from typing import List
from pydantic import BaseModel,AnyUrl

class AIInputUrl(BaseModel):
    image_url: AnyUrl
    label: str

class AIInputImage(BaseModel):
    image_base64: str
    label: str


class BBox(BaseModel):
    category: str
    bbox: List[int]
    score: float
    
class AIOuput(BaseModel):
    bbox_list: List[BBox]