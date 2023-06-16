from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import io
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from os import environ as env

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


# Load the trained model
model = load_model('path/to/model.h5')

# Define the target image size expected by the model
target_size = (224, 224)


def preprocess_image(image):
    img = image.resize(target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the image
    return img

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("HalloFish.html", {"request": request, "name": "John"})

# D:\Kampus Merdeka\Bangkit 2023\capstone-project\Website\templates\HalloFish.html

@app.get("/predict", response_class=HTMLResponse)
async def other_page(request: Request):
    return templates.TemplateResponse("uploadprediksi.html", {"request": request})

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    # Read the uploaded file
    image = Image.open(io.BytesIO(await file.read()))
    image_bytes = image.tobytes()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    # Preprocess the image
    img = preprocess_image(image)

    # Perform the prediction
    pred = model.predict(img)
    class_indices = {0: "WHITESPOT", 1: "BACTERIALFIN", 2: "REDSPOT", 3: "HEALTHY"}
    predicted_class = class_indices[np.argmax(pred)]
    confidence = np.max(pred) * 100
    print(predicted_class)

    return templates.TemplateResponse("result.html", {"request": request, "predicted_class": predicted_class, "confidence": confidence, "image": base64_image})
