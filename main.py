from fastapi import FastAPI , UploadFile , File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
app = FastAPI()

origins = ["http://localhost",
           "http://localhost:3000",
           ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL = load_model(r"C:\Users\User\Downloads\DataScience\Potatoe disease finder\saved_models\potatoes.h5" , compile=False)
CLASS_NAMES = ["Early Blight" , "Late Blight" , "Healthy"]

@app.get("/ping")
async def ping():
    return "hello i am alive"

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((256, 256))   # ✅ IMPORTANT (not 224)
    image = np.array(image) / 255.0
    return image
@app.post("/predict")
async def predict(
     file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = round(float(np.max(predictions[0])) * 100, 2)
    print(predicted_class , confidence)
    return{
    "class" : predicted_class,
    "confidence" : float(confidence)

}
    #to run this server use uvicorn
if __name__ == "__main__":
    uvicorn.run(app , host = "localhost" , port = 8000)

