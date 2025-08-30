# -*- coding: utf-8 -*- 
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import os

# ---------------------------- Set background ----------------------------
st.markdown("""
<style>
.stApp {
    background-image: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)),
                      url("https://t3.ftcdn.net/jpg/12/29/27/40/360_F_1229274084_rqWJRr06o1dOwZJxLdzFvqKLOXjthjK2.jpg");
    background-size: 95% auto;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: white;
}
h1 { color: #FFD700; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ---------------------------- Check required files ----------------------------
required_files = ["21P-CNN-Face Recognition.keras", "class_names.pkl"]
missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f"Missing required files: {missing_files}")
    st.info("Please upload these files to the same directory as app.py")
    for file in missing_files:
        st.write(f"• {file}")
    st.stop()

# ---------------------------- Load model and classes ----------------------------
@st.cache_resource
def load_model_and_classes():
    try:
        model = load_model("21P-CNN-Face Recognition.keras")
        with open("class_names.pkl", "rb") as f:
            class_names = pickle.load(f)
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, class_names = load_model_and_classes()
if model is None or class_names is None:
    st.stop()

# -------------------- Sidebar --------------------
st.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: rgba(0, 70, 30, 0.45);
    color: white;
}
[data-testid="stSidebar"] h1, h2, h3 { color: #00171F; }
::-webkit-scrollbar-thumb { background: #00cfff; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.sidebar.header("⚙️ Settings")
show_preprocessing = st.sidebar.checkbox("Show preprocessing steps", value=False)
show_confidence = st.sidebar.checkbox("Show confidence details", value=True)
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)


# Developer's intro
with st.sidebar.expander("👨‍💻 Developer's Intro"):
    st.markdown("- **Hi, I'm Rayyan Ahmed**")
    st.markdown("- **IBM Certified Advanced LLM FineTuner**")
    st.markdown("- **Google Certified Soft Skill Professional**")
    st.markdown("- **Hugging Face Certified: Fundamentals of LLMs**")
    st.markdown("- **Expert in EDA, ML, RL, ANN, CNN, CV, RNN, NLP, LLMs**")
    st.markdown("[💼 Visit LinkedIn](https://www.linkedin.com/in/rayyan-ahmed-504725321/)")

# Tech Stack
with st.sidebar.expander("🛠️ Tech Stack Used"):
    st.markdown("""
    - **Python Libraries:** Numpy, Pandas, Matplotlib, Seaborn  
    - **Machine Learning & AI:** Scikit-learn, TensorFlow, Keras  
    - **Serialization & Storage:** Pickle  
    - **Web App & UI:** Streamlit  
    - **Image Processing:** OpenCV, PIL (Pillow)  
    - **Version Control / Deployment:** Git, Streamlit Cloud
    """)

def preprocess_image(uploaded_file, show_steps=False):
    """Preprocess uploaded image to match model input shape (50, 37, 1)"""
    image = Image.open(uploaded_file).convert("L")  # grayscale
    img_array = np.array(image)
    resized = cv2.resize(img_array, (37, 50))
    normalized = resized / 255.0
    final_input = normalized.reshape(1, 50, 37, 1)

    if show_steps:
        st.write("**Preprocessing Steps:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("1. Original Image")
            st.image(image, width=150)
        with col2:
            st.write("2. Resized to 50x37")
            st.image(resized, width=150, channels="GRAY")
        with col3:
            st.write("3. Final Input Shape")
            st.write(final_input.shape)
            st.write(f"Value range: {final_input.min():.3f} to {final_input.max():.3f}")

    return final_input

def predict_with_confidence(model, img_array, class_names, show_details=False):
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = prediction[0][predicted_class]
    predicted_name = class_names[predicted_class]

    if show_details:
        st.write("**Top 5 Predictions:**")
        top5_indices = np.argsort(prediction[0])[::-1][:5]
        for i, idx in enumerate(top5_indices):
            name = class_names[idx]
            conf = prediction[0][idx]
            st.write(f"{i+1}. {name}: {conf:.4f} ({conf*100:.2f}%)")

    return predicted_name, confidence, prediction[0]

# ---------------------------- Streamlit UI ----------------------------
st.title("👤 Enhanced Face Recognition App - by Rayyan Ahmed")
st.write("Upload a face image to identify the person")

with st.expander("📋 Recognized People", expanded=False):
    st.write("The model can recognize the following people:")
    n = 4
    for i in range(0, len(class_names), n):
        line = " 🔹 ".join(class_names[i:i+n])
        st.write(f" 🔹 {line}")

uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1,1])
    with col1:
        st.subheader("📸 Uploaded Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image")
        st.write(f"Image size: {image.size}, mode: {image.mode}")

    with col2:
        st.subheader("🤖 Prediction Results")
        try:
            img_array = preprocess_image(uploaded_file, show_steps=show_preprocessing)
            predicted_name, confidence, _ = predict_with_confidence(
                model, img_array, class_names, show_details=show_confidence
            )

            if confidence >= confidence_threshold:
                st.success(f"Predicted: {predicted_name}")
            else:
                st.warning(f"Predicted: {predicted_name} (Low confidence)")

            st.write(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            st.progress(int(confidence * 100))

        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.write("Check if your model and image are compatible")

# ---------------------------- Model Info ----------------------------
with st.expander("ℹ️ Model Information", expanded=False):
    if model is not None:
        st.write(f"Model input shape: {model.input_shape}")
        st.write(f"Number of classes: {len(class_names)}")
        st.write(f"Number of layers: {len(model.layers)}")
        if st.button("Show model summary"):
            summary_list = []
            model.summary(print_fn=lambda x: summary_list.append(x))
            st.code("\n".join(summary_list), language='text')

# ---------------------------- Troubleshooting ----------------------------
with st.expander("🔧 Troubleshooting", expanded=False):
    st.write("""
    **Common Issues & Tips:**
    1. Upload a clear, front-facing photo.
    2. Face should be centered and well-lit.
    3. Enable 'Show preprocessing steps' to debug image processing.
    4. Adjust the confidence threshold if predictions seem low.
    5. Ensure the model was trained with enough images of all classes.
    6. Very similar-looking people may get confused.
    7. Lighting or resolution differences can reduce accuracy.
    """)
