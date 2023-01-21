import numpy as np
import cv2
import base64
import io
from PIL import Image

# base64 to numpy image
def base64_to_image(base64_string):
    imgdata = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(imgdata))
    return np.array(image)


