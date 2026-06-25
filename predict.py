from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
model = load_model(
    "models/pneumonia_model.h5"
)
image_path = "data/chest_xray/test/NORMAL/IM-0031-0001.jpeg"
img = Image.open(image_path).convert("RGB")

img = img.resize((224,224))

img = np.array(img)

img = img / 255.0

img = np.expand_dims(
    img,
    axis=0
)

prediction = model.predict(img)

if prediction[0][0] > 0.5:
    print("PNEUMONIA")
else:
    print("NORMAL")

