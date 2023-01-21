from typing import Optional
from PIL import Image
import cv2
import requests
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from decouple import config
import os

import uvicorn
from app.src.routers import inference

app = FastAPI()

def config_router(app):
	app.include_router(inference.router)
config_router(app)

if __name__ == "__main__":
	uvicorn.run(app, host="localhost", port=8000)
