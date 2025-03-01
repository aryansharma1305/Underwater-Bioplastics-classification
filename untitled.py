import streamlit as st
import torch
import detectron2
import cv2
import numpy as np
import tempfile
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo

# ✅ Set up Streamlit UI
st.set_page_config(page_title="Underwater Object Detection", layout="wide")
st.title("🌊 Underwater Object Detection with Mask R-CNN")

# ✅ Load model
@st.cache_resource()
def load_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16  # Update with actual number of classes
    cfg.MODEL.WEIGHTS = r"C:\Users\Dell\Downloads\maskrcnn_final_model2.pth"  # Update model path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Lower threshold for better detection
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    return DefaultPredictor(cfg)

predictor = load_model()

# ✅ Set Metadata
debris_classes = [
    "Mask", "can", "cellphone", "electronics", "gbottle", "glove",
    "metal", "misc", "net", "pbag", "pbottle", "plastic", "rod",
    "sunglasses", "tire"
]
MetadataCatalog.get("debris_train").thing_classes = debris_classes
metadata = MetadataCatalog.get("debris_train")

# ✅ File Upload for Image Input
uploaded_file = st.file_uploader("📷 Upload an underwater image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # ✅ Read Image from Upload
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        image_path = temp_file.name

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # ✅ Preserve original resolution (avoid resizing)
    original_height, original_width = image.shape[:2]
    if max(original_width, original_height) > 1200:  # Resize only if too large
        scale = 1200 / max(original_width, original_height)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

    # ✅ Run Inference
    outputs = predictor(image)

    # ✅ Visualize Results (With Label Overlap Reduction)
    v = Visualizer(
        image[:, :, ::-1],
        metadata=metadata,
        scale=1.2,  # Slightly increase scale for clarity
        instance_mode=ColorMode.IMAGE_BW  # Reduce background intensity
    )
    
    v._default_font_size = 14  # Reduce font size for less clutter
    v._default_line_width = 2  # Ensure bounding box lines are visible
    v._default_label_line_spacing = 10  # Increase spacing to reduce overlap

    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # ✅ Convert OpenCV image to RGB for Streamlit display
    result_image = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)

    # ✅ Display Results
    st.image(result_image, caption="🛠️ Segmented Image (Overlap Reduced)", use_container_width=True)

    # ✅ Debugging - Print detected classes
    detected_classes = outputs["instances"].pred_classes.tolist()
    detected_labels = [debris_classes[i] for i in detected_classes] if detected_classes else []
    st.write("🛑 **Detected Objects:**", detected_labels)

    # ✅ Clean up temp file
    os.remove(image_path)

st.success("✅ Ready for processing! Upload an image to see results.")
