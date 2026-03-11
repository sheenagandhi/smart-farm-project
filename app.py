import streamlit as st
import random
import time
import pandas as pd
import json
import numpy as np
import tensorflow as tf
from PIL import Image

# --- CONFIGURATION ---
st.set_page_config(page_title="Smart Farm IoT", layout="wide")
st.title("🌱 Smart Farming IoT & AI Dashboard")
st.markdown("### Real-time Monitoring & Crop Disease Detection")

# --- LOAD AI MODEL & CLASS NAMES ---
@st.cache_resource
def load_ai_doctor():
    # Load your teammate's trained model
    model = tf.keras.models.load_model('plant_disease_model.keras')
    
    # Load the dictionary that translates AI numbers to English names
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
        
    return model, class_names

try:
    disease_model, class_names = load_ai_doctor()
    ai_ready = True
except Exception as e:
    st.warning("⚠️ AI Model not found. Please ensure 'plant_disease_model.keras' and 'class_names.json' are uploaded.")
    ai_ready = False

# --- UI LAYOUT: TWO TABS ---
tab1, tab2 = st.tabs(["📊 Live Sensor Data", "🩺 AI Crop Doctor"])

# ==========================================
# TAB 1: SENSOR SIMULATION (The "IoT" part)
# ==========================================
with tab1:
    st.header("Field 1: Environmental Sensors")
    
    col1, col2, col3 = st.columns(3)
    moisture_metric = col1.empty()
    temp_metric = col2.empty()
    status_metric = col3.empty()
    
    if st.button("Start Sensor Simulation"):
        chart_data = pd.DataFrame(columns=["Temperature", "Moisture"])
        chart_placeholder = st.empty()

        for i in range(50): # Runs for 50 seconds
            temp = random.randint(20, 40)
            moisture = max(0, min(100, 100 - (temp * 1.5) + random.randint(-5, 5)))
            
            moisture_metric.metric("Soil Moisture", f"{int(moisture)}%")
            temp_metric.metric("Temperature", f"{temp}°C")
            
            if moisture < 30:
                status_metric.error("🚨 CRITICAL: Soil Dry. Pump ON.")
            else:
                status_metric.success("✅ Normal")
                
            new_data = pd.DataFrame({"Temperature": [temp], "Moisture": [moisture]})
            chart_data = pd.concat([chart_data, new_data], ignore_index=True)
            chart_placeholder.line_chart(chart_data)
            
            time.sleep(1)

# ==========================================
# TAB 2: AI DISEASE DETECTION
# ==========================================
with tab2:
    st.header("Upload a leaf image for diagnosis")
    
    uploaded_file = st.file_uploader("Choose a plant leaf image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and ai_ready:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Leaf', use_container_width=True)
        
        with st.spinner('AI is analyzing the leaf...'):
            # Format the image for the MobileNetV2 AI
            img_resized = image.resize((224, 224)) 
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = img_array / 255.0 
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = disease_model.predict(img_array)
            predicted_class_index = np.argmax(predictions)
            confidence_score = np.max(predictions) * 100
            
            predicted_disease = class_names[str(predicted_class_index)]
            
            # Results
            st.divider()
            if "healthy" in predicted_disease.lower():
                st.success(f"### Diagnosis: {predicted_disease.replace('_', ' ')}")
            else:
                st.error(f"### Diagnosis: {predicted_disease.replace('_', ' ')}")
                
            st.info(f"**AI Confidence:** {confidence_score:.2f}%")
