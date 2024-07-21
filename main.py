from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Initialize the FastAPI app
app = FastAPI()

# Load the pre-trained model
model = load_model('Fashion_MNIST.keras')

@app.get("/")
async def read_root():

    return {
        "message": "Welcome to the Fashion_MNIST digit classification API!",
        "instructions": {
            "POST /predict/": "Upload Fashion_MNIST digit (28x28 pixels) to get the predicted class."
        }
    }

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):

    # Convert the uploaded image to grayscale, resize to 28x28 pixels
    img = Image.open(file.file).convert('L').resize((28, 28))
    
    # Convert image to numpy array and normalize
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
    
    # Predict the class of the digit
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    
    # Return the predicted class as a JSON response
    return {"predicted_class": int(predicted_class)}