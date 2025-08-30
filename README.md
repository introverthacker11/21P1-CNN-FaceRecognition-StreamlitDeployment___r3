![CNN Face Recognition](https://ars.els-cdn.com/content/image/1-s2.0-S2665917423001368-gr6.jpg)

# 👤 CNN-Based Face Recognition Web App

A real-time Face Recognition Web Application built with TensorFlow & Keras. The model is trained on the LFW (Labeled Faces in the Wild) dataset and deployed using Streamlit. Users can upload a face image, and the app predicts the person with confidence scores.

Streamlit Link: ![streamlit link](https://cnn-facerecognition-appdeploymentr3-jj5vug49rqdwmqcgkbnah.streamlit.app/)

## 📂 Project Structure

├── app.py                         # Streamlit web application  
├── 21P-CNN-Face Recognition.keras # Trained CNN model  
├── class_names.pkl                # Serialized class labels  
├── requirements.txt               # Dependencies  
└── README.md                      # Project documentation  

## ⚙️ Installation

1. Clone the repository:

```bash

git clone https://github.com/your-username/face-recognition-app.git
cd face-recognition-app
```

2. Create a virtual environment (recommended):

```bash

python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
 ```

3. Install Dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Make sure 21P-CNN-Face Recognition.keras and class_names.pkl are in the same folder as app.py.

2. Run the app:

```bash
streamlit run app.py
```

3. Upload a face image (.jpg, .jpeg, .png) and view predictions.

## 🧠 Model Details:

- Dataset: LFW (Labeled Faces in the Wild)

- Input size: 50 × 37 grayscale

- Architecture:

  - Conv2D → MaxPooling2D

  - Conv2D → MaxPooling2D

  - Flatten → Dense (128)

  - Output Dense (softmax, number of classes)

- Optimizer: Adam

- Loss: Categorical Crossentropy

- Accuracy: ~XX% (replace with your test accuracy)

## 📸 Features: 

- ✅ Upload and recognize faces

- ✅ Show top-5 predictions with confidence

- ✅ Adjustable confidence threshold

- ✅ View preprocessing steps (resizing, normalization)

- ✅ Model summary and details available

- ✅ Clean UI with expandable sections

## 🛠️ Tech Stack:

- Python Libraries: Numpy, Pandas, Matplotlib, Seaborn

- Machine Learning: TensorFlow, Keras, Scikit-learn

- Image Processing: OpenCV, Pillow

- Web Framework: Streamlit

- Serialization: Pickle

---

X-X-X
