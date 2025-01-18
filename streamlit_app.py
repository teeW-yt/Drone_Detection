import streamlit as st
import cv2
import numpy as np
import torch
import os
import io

from PIL import Image
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

# -----------------------------
# 1. SETUP CONFIG AND PREDICTOR
# -----------------------------
@st.cache_resource  # so we don't reload model every time user uploads a new file
def load_model():
    cfg = get_cfg()
    cfg.MODEL.DEVICE = 'cpu'
    cfg.set_new_allowed(True)
    # Use the same config you used in training
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    )
    cfg.DATASETS.TRAIN = ("train_dataset",)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # update with your number of classes
    cfg.MODEL.WEIGHTS = os.path.join(".", "model_final.pth")  # path to your trained weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # confidence threshold
    predictor = DefaultPredictor(cfg)

    # If you have custom metadata from your DatasetCatalog
    # Here we assume "train_dataset" or "val_dataset" is registered.
    # For visualization, we just attach some generic metadata.
    # You can also skip this or use your real metadata object.
    metadata = MetadataCatalog.get("train_dataset")
    return predictor, metadata

predictor, metadata = load_model()

st.title("Drone Detection Demo with Detectron2")

# -------------------------------------------
# 2. IMAGE UPLOAD AND INFERENCE
# -------------------------------------------
st.header("Detect objects in an **image**")
uploaded_img = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    # Convert the uploaded file to an opencv image
    file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Run inference
    outputs = predictor(img)
    
    # Visualize results
    v = Visualizer(
        img[:, :, ::-1],
        metadata=metadata,
        scale=0.8,
        instance_mode=ColorMode.IMAGE
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    
    # Convert back to PIL/streamlit-compatible format
    result_img = out.get_image()[:, :, ::-1]
    st.image(result_img, channels="BGR", caption="Detections")

    # Optionally, show raw output: bounding boxes, classes, scores
    # st.write(outputs["instances"].to("cpu"))

# -------------------------------------------
# 3. VIDEO UPLOAD AND FRAME-BY-FRAME INFERENCE
# -------------------------------------------
st.header("Detect objects in a **video**")
uploaded_vid = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_vid is not None:
    # Save uploaded video to a temporary file on disk
    temp_input_video = "temp_input_video.mp4"
    with open(temp_input_video, "wb") as f:
        f.write(uploaded_vid.read())
    
    # Open video with OpenCV
    cap = cv2.VideoCapture(temp_input_video)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
    else:
        # Prepare to write out the processed video
        temp_output_video = "temp_output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_writer = cv2.VideoWriter(temp_output_video, fourcc, fps, (width, height))
        
        progress_bar = st.progress(0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0
        
        # Read and process frame-by-frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            outputs = predictor(frame)
            
            v = Visualizer(
                frame[:, :, ::-1],
                metadata=metadata,
                scale=0.5,
                instance_mode=ColorMode.IMAGE
            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            processed_frame = out.get_image()[:, :, ::-1]
            
            # Write processed frame to output video
            out_writer.write(processed_frame)
            
            frame_num += 1
            progress_bar.progress(min(frame_num / frame_count, 1.0))
        
        cap.release()
        out_writer.release()
        
        st.success("Video processing complete!")
        st.video(temp_output_video)
        
        # Cleanup temp files if desired
        # os.remove(temp_input_video)
        # os.remove(temp_output_video)
