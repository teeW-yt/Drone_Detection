# fastapi_endpoint.py
import io
import os
import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog


app = FastAPI()


def load_model():
    """
    Initialize the Detectron2 model (CPU by default).
    Returns (predictor, metadata).
    """
    cfg = get_cfg()
    cfg.MODEL.DEVICE = "cpu"   # or "mps" if on Apple Silicon
    cfg.set_new_allowed(True)

    # Load config you used in training
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    )
    # Adjust for your number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
    # Path to your trained weights
    cfg.MODEL.WEIGHTS = os.path.join(".", "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7

    predictor = DefaultPredictor(cfg)
    
    # Make sure "train_dataset" is registered with detectron2, or set manual classes:
    # MetadataCatalog.get("train_dataset").thing_classes = ["drone", "class2", "class3", ...]
    metadata = MetadataCatalog.get("train_dataset")
    return predictor, metadata


predictor, metadata = load_model()


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Upload an image and get back the **annotated/predicted image** as a response.
    """
    try:
        # Read file bytes
        file_bytes = await file.read()

        # Convert bytes to OpenCV format
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse(
                content={"error": "Could not decode image"},
                status_code=400
            )

        # Run inference
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")

        # Visualize bounding boxes
        v = Visualizer(
            img[:, :, ::-1],
            metadata=metadata,
            scale=0.8,
            instance_mode=ColorMode.IMAGE
        )
        out = v.draw_instance_predictions(instances)
        annotated_img = out.get_image()[:, :, ::-1]

        # Encode annotated_img as PNG (or JPEG)
        success, encoded_image = cv2.imencode(".png", annotated_img)
        if not success:
            return JSONResponse(
                content={"error": "Failed to encode image"},
                status_code=500
            )

        # Create a BytesIO stream from the encoded image
        return StreamingResponse(
            io.BytesIO(encoded_image.tobytes()),
            media_type="image/png"
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
