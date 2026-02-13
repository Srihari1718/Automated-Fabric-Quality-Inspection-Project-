def is_image_blurry(image_path, threshold=100):
    # Import OpenCV lazily so modules importing this file don't require cv2 at import time
    try:
        import cv2
    except Exception as e:
        raise ImportError(f"OpenCV (cv2) not available: {e}")

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold, round(laplacian_var, 2)
