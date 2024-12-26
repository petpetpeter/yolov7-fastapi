# YOLOV7-fastapi

Simple example of how to deploy YOLov7 model with FastAPI and Docker.

- clone repo
```
git clone https://github.com/petpetpeter/yolov7-fastapi.git
```
- start app locally
```
uvicorn app.main:app --reload
```
- try client
```
python app/webcam_client.py
```


## How to run a service with docker

1. build a docker image
```
docker build -t yolo:7.0 .
```
2. run a service
```
docker run -d --name yolorunner -p 8000:8000 yolo:7.0 
```

3. Test a service
```
curl -X 'POST' \
  'http://127.0.0.1:8000/ai/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "image_url": "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg",
  "label": "person"
}'
```
or visit http://localhost:8000/docs for swagger UI


4. Stop a service
```
docker stop yolorunner
```

5. Remove a service
```
docker rm yolorunner
```

