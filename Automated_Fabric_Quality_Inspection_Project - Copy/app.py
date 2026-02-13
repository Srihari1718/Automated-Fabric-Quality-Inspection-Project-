import streamlit as st
import os
from PIL import Image
import random

st.set_page_config(page_title="Fabric Quality Inspection", page_icon="🧵")

st.title("🧵 Automated Fabric Quality Inspection")
st.subheader("Deep Learning Based Defect Detection with Expert Validation")

def _demo_detection(image_path):
    """Return a fake detection result so the UI can be demoed without a trained model."""
    classes = ["hole", "stain"]
    picked = random.choice(classes)
    confidence = round(random.uniform(60, 99), 2)
    return {"status": "accepted", "defect": picked, "confidence": confidence}


uploaded_file = st.file_uploader("Upload Fabric Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    os.makedirs("temp", exist_ok=True)
    image_path = f"temp/{uploaded_file.name}"

    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    image = Image.open(image_path)
    st.image(image, caption="Uploaded Fabric Image", width=700)

    analyze = st.button("Analyze Fabric")
    demo_mode = st.checkbox("Use demo mode (no model required)")

    if analyze:
        with st.spinner("Analyzing..."):
            if demo_mode:
                result = _demo_detection(image_path)
            else:
                # Import the detector lazily so the app can load even if TensorFlow isn't installed
                try:
                    from src.predict import detect_defect
                except Exception:
                    # Keep the UI clean: show a single error message when detection backend isn't available
                    st.error("Defect detection currently unavailable.")
                    result = {"status": "rejected", "message": "Defect detection currently unavailable.", "blur_score": None}
                else:
                    # Call the detection function and handle its structured response
                    try:
                        result = detect_defect(image_path)
                    except Exception as e:
                        result = {"status": "rejected", "message": f"Detection failed: {e}", "blur_score": None}

        # Display results
        if result.get("status") == "rejected":
            st.error(result.get("message", "Analysis rejected"))
            if result.get("blur_score") is not None:
                st.info(f"Blur Score: {result['blur_score']}")
            # If missing model, offer helpful next steps
            msg = str(result.get("message", ""))
            if "Failed to load model" in msg or "No such file or directory" in msg:
                st.info("The trained model file is missing. Place the file at 'model/fabric_cnn.h5' or enable Demo mode.")
        else:
            status = result.get("status")
            if status == "accepted":
                st.success(f"Detected Defect: {result.get('defect','unknown').upper()}")
                st.write(f"Confidence: {result.get('confidence','N/A')} %")
                try:
                    from src.expert_validation import expert_validation
                    st.write(expert_validation(result.get("confidence", 0)))
                except Exception:
                    st.info("Expert validation unavailable.")
            elif status == "accepted_cv":
                st.success("Analysis completed (CV-based).")
                if result.get("blur_score") is not None:
                    st.write(f"Blur Score: {result.get('blur_score')}")
                st.write(f"Fabric Type (heuristic): {result.get('fabric_type','unknown')}" )
                st.write(f"Holes detected: {result.get('holes_count',0)}")
                st.write(f"Stains/spots detected: {result.get('stains_count',0)}")
                st.write(f"Misweave detected: {result.get('misweave', False)} (score: {result.get('misweave_score',0)})")
                st.write(f"Color variation score: {result.get('color_variation_score', 0)}")
                ann = result.get('annotated_image')
                if ann and os.path.exists(ann):
                    try:
                        ann_img = Image.open(ann)
                        st.image(ann_img, caption='Annotated image (holes=red, stains=blue, misweave=green)')
                    except Exception:
                        st.info(f"Annotated image saved at {ann}")
                # Expert validation still useful for CV results
                try:
                    from src.expert_validation import expert_validation
                    st.write(expert_validation(0))
                except Exception:
                    st.info("Expert validation unavailable.")
            else:
                # Unknown accepted status - show raw result
                st.write(result)
