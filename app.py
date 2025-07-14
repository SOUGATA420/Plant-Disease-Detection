import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import io
import base64
import os

# Load the trained model
model = tf.keras.models.load_model('models/final_model.h5')

# Define class labels
labels = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", 
    "Apple___healthy", "Background_without_leaves", "Blueberry___healthy", 
    "Cherry___Powdery_mildew", "Cherry___healthy", 
    "Corn___Cercospora_leaf_spot Gray_leaf_spot", "Corn___Common_rust", 
    "Corn___Northern_Leaf_Blight", "Corn___healthy", "Grape___Black_rot", 
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", 
    "Grape___healthy", "Orange___Haunglongbing_(Citrus_greening)", 
    "Peach___Bacterial_spot", "Peach___healthy", "Pepper,_bell___Bacterial_spot", 
    "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___Late_blight", 
    "Potato___healthy", "Raspberry___healthy", "Soybean___healthy", 
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy", 
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", 
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", 
    "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", 
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", 
    "Tomato___healthy"
]

# Set Streamlit page config
st.set_page_config(page_title="🌱 Plant Disease Detector", layout="wide")

# Language selection
lang = st.sidebar.selectbox("🌐 Choose Language", ["English", "Bengali", "Hindi"])

# Language dictionary
translations = {
    "English": {
        "title": "Plant Disease Detection",
        "desc": "Upload a plant leaf image and get instant predictions with confidence scores.",
        "input": "Choose Input Method",
        "upload": "Upload an image",
        "photo": "Take a picture",
        "paste": "Paste Base64 image string",
        "prediction": "Prediction Result",
        "top5": "Top 5 Predictions",
        "history": "Recent Predictions",
        "tips": "Tips for Better Results",
        "tip_text": "- Use a high-quality image focused on one leaf\n- Avoid blur, glare, or background noise\n- Capture images in natural light\n- Upload a .jpg or .png format image"
    },
    "Bengali": {
        "title": "গাছের রোগ শনাক্তকরণ",
        "desc": "একটি গাছের পাতার ছবি আপলোড করুন এবং আত্মবিশ্বাস সহকারে পূর্বাভাস পান।",
        "input": "ইনপুট পদ্ধতি নির্বাচন করুন",
        "upload": "একটি ছবি আপলোড করুন",
        "photo": "ছবি তুলুন",
        "paste": "Base64 ছবি স্ট্রিং পেস্ট করুন",
        "prediction": "পূর্বাভাসের ফলাফল",
        "top5": "সেরা ৫টি পূর্বাভাস",
        "history": "সাম্প্রতিক পূর্বাভাস",
        "tips": "ভাল ফলাফলের জন্য টিপস",
        "tip_text": "- একটি উচ্চমানের ছবি ব্যবহার করুন\n- ঝাপসা বা আলোর প্রতিফলন এড়ান\n- প্রাকৃতিক আলোতে ছবি তুলুন\n- .jpg বা .png ফরম্যাটে আপলোড করুন"
    },
    "Hindi": {
        "title": "पौधों के रोग की पहचान",
        "desc": "पत्ते की तस्वीर अपलोड करें और विश्वास के साथ पूर्वानुमान प्राप्त करें।",
        "input": "इनपुट विधि चुनें",
        "upload": "चित्र अपलोड करें",
        "photo": "फोटो लें",
        "paste": "Base64 छवि स्ट्रिंग पेस्ट करें",
        "prediction": "पूर्वानुमान परिणाम",
        "top5": "शीर्ष 5 भविष्यवाणियाँ",
        "history": "हाल की भविष्यवाणियाँ",
        "tips": "बेहतर परिणामों के लिए सुझाव",
        "tip_text": "- उच्च गुणवत्ता वाली छवि का उपयोग करें\n- धुंधली या चमकदार पृष्ठभूमि से बचें\n- प्राकृतिक प्रकाश में चित्र लें\n- .jpg या .png प्रारूप में अपलोड करें"
    }
}

# Get language strings
T = translations[lang]

# Page header and description
st.markdown(f"""
    <div style="text-align:center;">
        <h1 style="color:#2E8B57;">🌿 {T['title']}</h1>
        <p style="font-size:18px;">{T['desc']}</p>
    </div>
""", unsafe_allow_html=True)

# Preprocess image
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Select input method
method = st.radio(T['input'], ["📁 " + T['upload'], "📷 " + T['photo'], "🖋 " + T['paste']], horizontal=True)

image = None

if method == "📁 " + T['upload']:
    uploaded_file = st.file_uploader(T['upload'], type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

elif method == "📷 " + T['photo']:
    camera_image = st.camera_input(T['photo'])
    if camera_image:
        image = Image.open(camera_image)

elif method == "🖋 " + T['paste']:
    base64_input = st.text_area(T['paste'])
    try:
        if base64_input:
            decoded = base64.b64decode(base64_input)
            image = Image.open(io.BytesIO(decoded))
    except Exception as e:
        st.error("Invalid Base64 input. Please check and try again.")

# Two-column layout
col1, col2 = st.columns([1, 2])

if image is not None:
    col1.image(image, caption="Uploaded Leaf", use_column_width=True)

    processed = preprocess_image(image)
    prediction = model.predict(processed)

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    col2.subheader(f"🩺 {T['prediction']}")
    col2.markdown(f"<h3 style='color:#006400;'>✅ {labels[class_index]}</h3>", unsafe_allow_html=True)
    col2.markdown(f"<p style='font-size:16px;'>Confidence: <strong>{confidence*100:.2f}%</strong></p>", unsafe_allow_html=True)

    top_5_indices = prediction[0].argsort()[-5:][::-1]
    top_5 = {labels[i]: float(prediction[0][i]) for i in top_5_indices}
    top_5_df = pd.DataFrame.from_dict(top_5, orient='index', columns=['Probability'])
    col2.markdown(f"### 🔍 {T['top5']}")
    col2.bar_chart(top_5_df)

    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append((labels[class_index], round(confidence * 100, 2)))

# Sidebar: prediction history
with st.sidebar:
    st.markdown(f"### 🕓 {T['history']}")
    if 'history' in st.session_state:
        for i, (label, conf) in enumerate(st.session_state.history[-5:], 1):
            st.markdown(f"**{i}. {label}** — {conf}%")

    st.markdown("---")
    with st.expander("📌 " + T['tips']):
        st.write(T['tip_text'])

    st.info("This app uses a deep learning CNN model trained on augmented plant disease data.")