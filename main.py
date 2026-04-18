from fastapi import FastAPI, UploadFile, File
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load model
model = tf.keras.models.load_model("my_model.h5")

@app.get("/")
def home():
    return {"message": "MNIST API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("L")  # grayscale
    image = image.resize((28, 28))

    img_array = np.array(image)

    # 👇 এখানে add করবে (IMPORTANT)
    img_array = 255 - img_array

    # (OPTIONAL) invert যদি দরকার হয়
    # img_array = 255 - img_array

    # ✅ Normalize (same as training)
    img_array = img_array / 255.0

    # ✅ Flatten (VERY IMPORTANT)
    img_array = img_array.reshape(1, 784)

    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction))

    return {"prediction": predicted_class}