import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load your trained model
model = load_model("mobilenet_model.keras")

# Define class labels (same as yours)
class_labels = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy"
]

# Define prediction function
def predict_disease(image):
    image = image.resize((224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = round(np.max(prediction) * 100, 2)

    return f"🧬 Predicted: {predicted_class}\n🎯 Confidence: {confidence}%"

# Create Gradio interface
interface = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="🌾 AgroAI - Plant Disease Detector",
    description="Upload a leaf image to identify the crop disease using your trained model."
)

# Launch the interface
interface.launch()
