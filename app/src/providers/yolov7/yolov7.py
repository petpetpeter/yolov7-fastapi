import argparse
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
from numpy import random

import onnxruntime as ort

import logging as log

from app.src.providers.yolov7.utils import letterbox

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = f"{CURRENT_DIR}/assets/yolov7-tiny.onnx"
CLASS_NAME_PATH = f"{CURRENT_DIR}/assets/coco.names"

class YOLOV7():
    def __init__(self, 
            model_weights = MODEL_PATH, 
            names = CLASS_NAME_PATH,
            imgsz = (640, 640), 
            device = "auto"):
        self.names = self.load_classes(names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        self.imgsz = imgsz
        
        # Load model
        if device == "auto":
            if ort.get_device():
                self.sess = ort.InferenceSession(f"{model_weights}", providers=["CUDAExecutionProvider"])
                log.warning(f"Using GPU")
            else:
                self.sess = ort.InferenceSession(f"{model_weights}", providers=["CPUExecutionProvider"])
                log.warning("Using CPU")
        else:
            self.sess = ort.InferenceSession(f"{model_weights}", providers=["CPUExecutionProvider"])
        self.outname = [i.name for i in self.sess.get_outputs()]
        self.inname = [i.name for i in self.sess.get_inputs()]
        ini_output = self.sess.run(self.outname, {self.inname[0]: np.random.random((1, 3, self.imgsz[0], self.imgsz[1])).astype(np.float32)})[0]
    
    def load_classes(self,path):
        # Loads *.names file at 'path'
        with open(path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as last line)
        
    def detect(self, bgr_img, threshold = 0.2):
        #result
        bboxDict = dict((name,[]) for name in self.names)
        #preprocess
        origin_image = bgr_img.copy()
        image = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        image,ratio,dwdh = letterbox(image,auto=False)
        image = image.transpose((2,0,1))
        image = np.expand_dims(image,0)
        image = image.astype(np.float32)
        image /= 255
        inputs = {self.inname[0]: image}
        # Prediction
        outputs = self.sess.run(self.outname, inputs)
        #print(np.array(outputs[0]).shape)
        #Postprocess
        for batch in outputs:
            for result in batch:
                batch_id,x0,y0,x1,y1,cls_id,score = result
                box = np.array([x0,y0,x1,y1])
                box -= np.array(dwdh*2)
                box /= ratio
                box = box.round().astype(np.int32).tolist()
                cls_id = int(cls_id)
                score = round(float(score),3)
                #add result to bboxDict
                label = self.names[cls_id]
                bboxDict[label].append((box[0], box[1],box[2], box[3],score))
                #render
                name = self.names[cls_id]
                color = self.colors[cls_id]
                name += ' '+str(score)
                cv2.rectangle(origin_image,box[:2],box[2:],color,2)
                cv2.putText(origin_image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)
        return origin_image, bboxDict


