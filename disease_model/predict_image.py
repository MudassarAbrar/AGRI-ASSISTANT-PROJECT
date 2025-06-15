import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model
model = load_model("plant_disease_model.h5")  # or .keras

# Class labels (update as per your classes)
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

img_path = r"C:\Users\Mudassir\OneDrive\Desktop\AGRI-ASSISTANT\data\PlantVillage\Potato___Late_blight\0b2bdc8e-90fd-4bb4-bedb-485502fe8a96___RS_LB 4906.JPG" 
def predict_image(image_path):
    img = Image.open(image_path).resize((224, 224))  # Resize to match model input
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)
    print("Predicted class:", class_labels[np.argmax(prediction)])

    return predicted_class, round(confidence * 100, 2)

# Call the prediction function
if __name__ == "__main__":
    predicted_class, confidence = predict_image(img_path)
    print(f"\n✅ Prediction: {predicted_class}")
    print(f"🔍 Confidence: {confidence}%")
