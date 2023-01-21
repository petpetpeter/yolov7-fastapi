import sys
from app.src.providers.yolov7.yolov7 import YOLOV7
import cv2
from imread_from_url import imread_from_url

model_path = "src/providers/yolov7/assets/yolov7-tiny.onnx"
classes_path = "src/providers/yolov7/assets/coco.names"

# Initialize YOLOv7 object detector
yolov7 = YOLOV7(model_path,classes_path) 

img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
img = imread_from_url(img_url)

# Detect Objects
result_image,bbox_dict = yolov7.detect(img)