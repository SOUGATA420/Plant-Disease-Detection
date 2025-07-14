import streamlit as st
from PIL import Image
import base64
import io
import pandas as pd
from src.pipeline import PlantDiseasePipeline

# Load the pipeline
pipeline = PlantDiseasePipeline()

# Language selection
lang = st.sidebar.selectbox("🌐 Choose Language", ["English", "Bengali", "Hindi"])

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
        "title": "গাছের রোগ শনাকতকরণ",
        "desc": "একটি গাছের পাতার ছবি আপলোড করুন এবং আত্মবিশ্বাস সহকারে পূর্বভাস পান।",
        "input": "ইনপুট পদ্ধতি নির্বাচন করুন",
        "upload": "একটি ছবি আপলোড করুন",
        "photo": "ছবি তুলুন",
        "paste": "Base64 ছবি স্ট্রিং পেস্ট করুন",
        "prediction": "পূর্বভাসের ফলাফল",
        "top5": "শীর্ষ ৫টি পূর্বভাস",
        "history": "সাম্প্রতিক পূর্বভাস",
        "tips": "ভাল ফলাফলের জন্য টিপস",
        "tip_text": "- একটি উচ্চমান মানের ছবি ব্যবহার করুন\n- প্রতিফলন অবর পৃষ্ঠভূমি পৃষ্ঠম এড়ান\n- প্রাকৃতিক আলোয়ে ছবি তুলুন\n- .jpg অথবা .png ফরম্যাটে আপলোড করুন"
    },
    "Hindi": {
        "title": "पौधों के रोग की पहचान",
        "desc": "पत्ते की तस्वीर अपलोड करें और विश्वास के साथ पूर्वानुमान प्राप्त करें।",
        "input": "इनपुट विधि चुनें",
        "upload": "चित्र अपलोड करें",
        "photo": "फोटो लें",
        "paste": "Base64 छवि स्ट्रिंग चैपकार करें",
        "prediction": "पूर्वानुमान परिणाम",
        "top5": "शीर्ष 5 भविष्यान्यां",
        "history": "हाल की भविष्यान्यां",
        "tips": "बेटर परिणामों के लिए सुझाव",
        "tip_text": "- उच्च गुणवत्ता की छवि का उपयोग करें\n- धुंधली या चमकदार पृष्ठभूमि पृष्ठम से बचें\n- प्राकृतिक चैर में छवि खींचें\n- .jpg या .png फार्मैट में अपलोड करें"
    }
}

T = translations[lang]

st.markdown(f"""
    <div style="text-align:center;">
        <h1 style="color:#2E8B57;">{T['title']}</h1>
        <p style="font-size:18px;">{T['desc']}</p>
    </div>
""", unsafe_allow_html=True)

# Input method
method = st.radio(T['input'], [f"📁 {T['upload']}", f"📷 {T['photo']}", f"🖋 {T['paste']}"], horizontal=True)

image = None
if method.startswith("📁"):
    uploaded_file = st.file_uploader(T['upload'], type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)

elif method.startswith("📷"):
    camera_image = st.camera_input(T['photo'])
    if camera_image:
        image = Image.open(camera_image)

elif method.startswith("🖋"):
    base64_input = st.text_area(T['paste'])
    if base64_input:
        try:
            decoded = base64.b64decode(base64_input)
            image = Image.open(io.BytesIO(decoded))
        except Exception:
            st.error("Invalid Base64 image string.")
col1, col2 = st.columns([1, 2])

if image is not None:
    col1.image(image, caption="Uploaded Leaf", use_column_width=True)

    result = pipeline.run(image)

    col2.subheader(f"🩺 {T['prediction']}")
    col2.markdown(f"<h3 style='color:#006400;'>✅ {result['label']}</h3>", unsafe_allow_html=True)
    col2.markdown(f"<p style='font-size:16px;'>Confidence: <strong>{result['confidence']*100:.2f}%</strong></p>", unsafe_allow_html=True)

    top_5_df = pd.DataFrame.from_dict(result['top_5'], orient='index', columns=['Probability'])
    col2.markdown(f"### 🔍 {T['top5']}")
    col2.bar_chart(top_5_df)

    # Save prediction history
    if 'history' not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append((result['label'], round(result['confidence'] * 100, 2)))
with st.sidebar:
    st.markdown(f"### 🕓 {T['history']}")
    if 'history' in st.session_state:
        for i, (label, conf) in enumerate(st.session_state.history[-5:], 1):
            st.markdown(f"**{i}. {label}** — {conf}%")
    st.markdown("---")
    with st.expander("📌 " + T['tips']):
        st.write(T['tip_text'])
    st.info("This app uses a deep learning CNN model trained on augmented plant disease data.")