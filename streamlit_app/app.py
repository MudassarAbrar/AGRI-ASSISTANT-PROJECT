import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests
import io
import openai
from dotenv import load_dotenv
import os

load_dotenv()


# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="AgroAI - Plant Assistant", layout="wide")
openai_api_key = os.getenv("OPENAI_API_KEY")  
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY") 
MODEL_PATH = r"C:\Users\Mudassir\OneDrive\Desktop\AGRI-ASSISTANT\disease_model\plant_disease_model.keras"

# ---------------------- LOAD MODEL ----------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

CLASS_NAMES =  [
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
# ---------------------- IMAGE PREDICTION ----------------------
def predict_disease(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)[0]
    idx = np.argmax(predictions)
    return CLASS_NAMES[idx], float(predictions[idx] * 100)

# ---------------------- WEATHER FETCH ----------------------
def get_weather(location):
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={location}"
    res = requests.get(url)

    if res.status_code == 200:
        data = res.json()
        return {
            "temp": data["current"]["temp_c"],
            "humidity": data["current"]["humidity"],
            "condition": data["current"]["condition"]["text"]
        }
    else:
        st.error(f"⚠️ API returned {res.status_code}: {res.text}") 
        return None


# ---------------------- GPT SUGGESTION ----------------------



def get_cure_suggestion(disease, weather):
    system_message = {
        "role": "system",
        "content": (
            "You are a professional agricultural assistant. "
            "Given a specific plant disease and current weather conditions (temperature, humidity, and weather description), "
            "you must give very specific remedies, pesticides (brand/generic names), and farming methods to help the farmer recover the crop. "
            "Specific **fertilizer recommendations** based on the plant condition and weather"
            "Be practical, not generic. Mention dosages if possible. Tailor the solution to the weather conditions."
        )
    }

    user_message = {
        "role": "user",
        "content": (
            f"The detected plant disease is '{disease}'. "
            f"The current weather is: Temperature: {weather['temp']}°C, "
            f"Humidity: {weather['humidity']}%, "
            f"Condition: {weather['condition']}. "
            "What should the farmer do?"
            "Give Specific **fertilizer recommendations** based on the plant condition and weather"
        )
    }

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[system_message, user_message]
    )

    return response.choices[0].message.content.strip()
# ---------------------- UI ----------------------
st.title("🌿 AgroAI - Plant Disease & Weather Assistant")
st.markdown("Upload a clear image of a plant leaf and enter your location. We’ll detect diseases and suggest remedies.")

col1, col2 = st.columns(2)

# Image input
with col1:
    uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Leaf", use_container_width=True)

# Location input
with col2:
    location = st.text_input("Enter Your City (e.g., Lahore)")

# Submit button
if uploaded_file and location:
    with st.spinner("Analyzing image and fetching data..."):
        disease, confidence = predict_disease(image)
        weather = get_weather(location)

        if weather:
            st.success(f"✅ Disease: {disease} (Confidence: {confidence:.2f}%)")
            st.info(f"🌤️ Weather in {location}: {weather['temp']}°C, {weather['humidity']}% Humidity, {weather['condition']}")

            with st.spinner("Generating care suggestions with ChatGPT..."):
                suggestion = get_cure_suggestion(disease, weather)
                st.markdown("---")
                st.subheader("🌱 Suggested Remedies")
                st.write(suggestion)
        else:
            st.error("⚠️ Could not fetch weather data. Please check the location name.")
else:
    st.warning("Please upload an image and enter your location.")
