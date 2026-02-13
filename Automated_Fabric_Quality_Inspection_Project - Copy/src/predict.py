MODEL_PATH = "model/fabric_cnn.h5"
IMG_SIZE = 224
CLASSES = ["hole", "stain"]

# Lazy-load heavy libraries and the model so the Streamlit UI can import this module
# even if TensorFlow/OpenCV are not installed. detect_defect returns a user-friendly
# rejection message when dependencies or the model are missing.
_loaded_model = None

def _load_model():
    global _loaded_model
    if _loaded_model is not None:
        return _loaded_model

    try:
        from tensorflow.keras.models import load_model
    except Exception as e:
        raise ImportError(f"TensorFlow not available: {e}")

    try:
        _loaded_model = load_model(MODEL_PATH)
        return _loaded_model
    except Exception as e:
        raise FileNotFoundError(f"Failed to load model '{MODEL_PATH}': {e}")

def detect_defect(image_path):
    # Import the image quality checker lazily to avoid requiring OpenCV at module import
    try:
        from src.image_quality_check import is_image_blurry
    except Exception as e:
        return {
            "status": "rejected",
            "message": f"Image quality check unavailable: {e}",
            "blur_score": None
        }

    # Check blur first (this uses OpenCV internally; handle its absence)
    try:
        blurry, blur_score = is_image_blurry(image_path)
    except Exception as e:
        return {
            "status": "rejected",
            "message": f"Unable to check image quality: {e}",
            "blur_score": None
        }

    if blurry:
        return {
            "status": "rejected",
            "message": "Image is blurred or fabric texture is unclear. Please upload a clear image.",
            "blur_score": blur_score
        }

    # Now attempt to import cv2 and numpy and the model
    try:
        import cv2
        import numpy as np
    except Exception as e:
        return {
            "status": "rejected",
            "message": f"Required image libraries missing: {e}. Install requirements.",
            "blur_score": blur_score
        }

    try:
        model = _load_model()
    except ImportError as e:
        return {
            "status": "rejected",
            "message": str(e),
            "blur_score": blur_score
        }
    except FileNotFoundError as e:
        # If the trained model file is missing, fall back to classical CV analysis
        try:
            return analyze_with_cv(image_path, blur_score=blur_score)
        except Exception as ex:
            return {
                "status": "rejected",
                "message": f"Model missing and CV fallback failed: {ex}",
                "blur_score": blur_score
            }

    # Preprocess and predict
    try:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        prediction = model.predict(image)
        class_index = int(np.argmax(prediction))
        confidence = float(prediction[0][class_index] * 100)

        return {
            "status": "accepted",
            "defect": CLASSES[class_index],
            "confidence": round(confidence, 2)
        }
    except Exception as e:
        return {
            "status": "rejected",
            "message": f"Prediction failed: {e}",
            "blur_score": blur_score
        }


def analyze_with_cv(image_path, blur_score=None):
    import cv2
    import numpy as np
    import os

    img = cv2.imread(image_path)
    if img is None:
        return {"status": "rejected", "message": "Unable to read image for CV analysis", "blur_score": blur_score}

    h, w = img.shape[:2]

    # Compute blur score if not provided
    if blur_score is None:
        try:
            from src.image_quality_check import is_image_blurry
            blurry, blur_score = is_image_blurry(image_path)
        except Exception:
            blur_score = None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Fabric type heuristic: edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean()
    if edge_density < 0.01:
        fabric_type = "smooth/plain"
    elif edge_density < 0.03:
        fabric_type = "knit-like"
    else:
        fabric_type = "woven/textured"

    # Hole detection: find dark circular/irregular regions
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    _, th = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # holes are dark, so invert threshold
    th_inv = cv2.bitwise_not(th)
    # Morphological opening to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    clean = cv2.morphologyEx(th_inv, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hole_contours = [c for c in contours if cv2.contourArea(c) > max(50, (w*h)*0.0001)]
    holes_count = len(hole_contours)

    # Stain detection: color deviation from local median using Lab color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # compute median color per channel
    med_l = np.median(l)
    med_a = np.median(a)
    med_b = np.median(b)
    color_dist = np.sqrt((l - med_l)**2 + (a - med_a)**2 + (b - med_b)**2)
    color_mask = (color_dist > 25).astype('uint8') * 255
    # clean mask
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel2, iterations=1)
    contours_s, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stain_contours = [c for c in contours_s if cv2.contourArea(c) > max(200, (w*h)*0.0005)]
    stains_count = len(stain_contours)

    # Misweave heuristic: measure local variance across grid; large local deviation indicates misweave
    grid_size = 32
    variances = []
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            patch = gray[y:y+grid_size, x:x+grid_size]
            if patch.size == 0:
                continue
            variances.append(patch.var())
    var_arr = np.array(variances)
    if var_arr.size == 0:
        misweave = False
        misweave_score = 0.0
    else:
        med = np.median(var_arr)
        # fraction of blocks that deviate strongly
        outlier_frac = np.mean(var_arr > (med * 2.5 + 1e-6))
        misweave = outlier_frac > 0.05
        misweave_score = float(outlier_frac)

    # Color variation score: cluster colors and see distribution imbalance
    reshaped = img.reshape((-1,3)).astype(np.float32)
    # use kmeans with k=2 for speed
    try:
        _, labels, centers = cv2.kmeans(reshaped, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), 3, cv2.KMEANS_RANDOM_CENTERS)
        counts = np.bincount(labels.flatten(), minlength=2)
        prop = counts / counts.sum()
        color_variation_score = float(1.0 - np.max(prop))  # closer to 0 means uniform, higher means variation
    except Exception:
        color_variation_score = 0.0

    # Annotate image: draw holes in red, stains in blue, misweave regions with green rectangles
    annotated = img.copy()
    for c in hole_contours:
        x,y,wc,hc = cv2.boundingRect(c)
        cv2.rectangle(annotated, (x,y), (x+wc, y+hc), (0,0,255), 2)
    for c in stain_contours:
        x,y,wc,hc = cv2.boundingRect(c)
        cv2.rectangle(annotated, (x,y), (x+wc, y+hc), (255,0,0), 2)
    if misweave:
        # mark top outlier blocks
        idxs = np.where(var_arr > (med * 2.5 + 1e-6))[0]
        # compute grid coordinates for each idx
        cnt = 0
        for idx in idxs[:50]:
            gy = (idx // ((w + grid_size -1)//grid_size)) * grid_size
            gx = (idx % ((w + grid_size -1)//grid_size)) * grid_size
            cv2.rectangle(annotated, (gx, gy), (min(gx+grid_size, w-1), min(gy+grid_size, h-1)), (0,255,0), 1)
            cnt += 1

    # Save annotated image to temp
    os.makedirs("temp", exist_ok=True)
    base = os.path.basename(image_path)
    annotated_path = os.path.join("temp", f"annotated_{base}")
    cv2.imwrite(annotated_path, annotated)

    return {
        "status": "accepted_cv",
        "blur_score": blur_score,
        "fabric_type": fabric_type,
        "holes_count": int(holes_count),
        "stains_count": int(stains_count),
        "misweave": bool(misweave),
        "misweave_score": misweave_score,
        "color_variation_score": round(color_variation_score, 3),
        "annotated_image": annotated_path
    }
