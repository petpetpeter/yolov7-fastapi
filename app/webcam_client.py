# python client for run inference on web server

import requests
import json
import cv2
import numpy as np
import base64
import time
from src.models.ai_model import AIInputImage

url = "http://localhost:8000/ai/predict/image"

#loop webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    # encode image as jpeg
    _, img_encoded = cv2.imencode('.jpg', frame)
    # convert to base64 encoding and show start of data
    img_base64 = base64.b64encode(img_encoded)
    payload = AIInputImage(image_base64=img_base64, label="person")
    
    # send http request with image and receive response
    response = requests.post(url, json=payload.dict())

    # decode response
    result = json.loads(response.text)
    
    #print(result)
    # draw bbox
    for bbox in result["bbox_list"]:
        cv2.rectangle(frame, (bbox["bbox"][0], bbox["bbox"][1]), (bbox["bbox"][2], bbox["bbox"][3]), (0, 255, 0), 2)
        cv2.putText(frame, bbox["category"] + " " + str(bbox["score"]), (bbox["bbox"][0], bbox["bbox"][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # show image
    cv2.imshow("result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break