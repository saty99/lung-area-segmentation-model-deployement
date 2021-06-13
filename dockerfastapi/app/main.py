# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 12:50:11 2021

@author: saty
"""
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import io
import uvicorn
import numpy as np
import nest_asyncio
from enum import Enum
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from starlette.responses import StreamingResponse
from keras.models import *

# Assign an instance of the FastAPI class to the variable "app".
# You will interact with your api using this instance.
app = FastAPI(title='Deploying a Lung Area Segmentation Model with FastAPI')

# Load the model (Weight file folder in the repo contains the model)
model = load_model('D:/Lung_Segmentation_Deployment/Weights file/model.h5', compile=False)


# By using @app.get("/") you are allowing the GET method to work for the / endpoint.
@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:8000/docs."


# This endpoint handles all the logic necessary for the object detection to work.
# It requires the desired model and the image in which to perform object detection.
@app.post("/predict") 
def prediction(file: UploadFile = File(...)):

    # 1. VALIDATE INPUT FILE
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
    
    # 2. TRANSFORM RAW IMAGE INTO CV2 image
    
    # Read image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())
    
    # Start the stream from the beginning (position zero)
    image_stream.seek(0)
    
    # Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    
    # Decode the numpy array as an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img=cv2.resize(image,(512,512))

    x_im = cv2.resize(image,(512,512))[:,:,0]
    
    # 3. RUN LUNG SEGMENTATION DETECTION MODEL
    
    # Run lung segmentation model
    op = model.predict((x_im.reshape(1, 512, 512, 1)-127.0)/127.0)
    mask=op[0,:,:,0]
    mask = (mask > 0.5).astype(np.uint8)
    
    # Create image that includes bounding boxes and labels
    final_image=cv2.bitwise_and(img, img, mask=mask.astype(np.uint8))
    
    # 4. STREAM THE RESPONSE BACK TO THE CLIENT
    
    res, im_jpeg = cv2.imencode(".jpeg", final_image)
    
    # Open the saved image for reading in binary mode
    #file_image = open(f'images_uploaded/{filename}', mode="rb")
    
    # Return the image as a stream specifying media type
    return StreamingResponse(io.BytesIO(im_jpeg.tobytes()), media_type="image/jpeg")


