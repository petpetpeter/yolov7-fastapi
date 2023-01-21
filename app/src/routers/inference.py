from fastapi import APIRouter,HTTPException
from app.src.models.ai_model import AIInputUrl,AIInputImage, AIOuput, BBox
from app.src.utils import image_utils
from app.src.providers.yolov7.yolov7 import YOLOV7
from imread_from_url import imread_from_url
import time
import os

router = APIRouter()
model = YOLOV7()

@router.get("/ai", tags=["AI"])
async def hello_world():
    return {"message": "Hello World"}


@router.post("/ai/predict/url", tags=["AI"])
def predict(payload: AIInputUrl):
    class_names = model.names
    if payload.label not in class_names:
        raise HTTPException(status_code=400, detail="Invalid label")
    result = AIOuput(bbox_list=[])
    image = imread_from_url(payload.AIInputUrl)
    origin_image,bboxDict = model.detect(image)
    for detection in bboxDict[payload.label]:
        bbox = BBox(category=payload.label, bbox=detection[0:4], score=detection[4])
        result.bbox_list.append(bbox)
    return result

@router.post("/ai/predict/image", tags=["AI"])
def predict(payload: AIInputImage):
    class_names = model.names
    if payload.label not in class_names:
        raise HTTPException(status_code=400, detail="Invalid label")
    result = AIOuput(bbox_list=[])
    image = image_utils.base64_to_image(payload.image_base64)
    origin_image,bboxDict = model.detect(image)
    for detection in bboxDict[payload.label]:
        bbox = BBox(category=payload.label, bbox=detection[0:4], score=detection[4])
        result.bbox_list.append(bbox)
    return result