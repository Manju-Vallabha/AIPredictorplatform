import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

def predict_brain_tumor(image_path, model_path='breast_cancer.h5'):
    # Load the trained model
    model = load_model(model_path)

    # Preprocess the input image
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize the image

    # Predict class probabilities
    predictions = model.predict(img_array)

    # Decode predictions (optional)
    # Assuming classes are ['brain_tumor', 'brain_glioma', 'brain_menin']
    class_labels = ['breast_bengin','breast_malignant']
    predicted_class = class_labels[np.argmax(predictions)]

    # Return the predicted class probabilities or labels
    return predicted_class, predictions[0]

# Streamlit app
st.title("Breast Cancer Classification")
with st.sidebar:
    st.info("Expreience the power of AI in detecting breast cancer")
    c = st.selectbox("Capture Image",("Camera","Upload Image"))
    if c == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if c == "Camera":
    uploaded_file = st.camera_input("Capture Image")
if uploaded_file is not None :
    if c == "Upload Image":
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    if st.button("Predict"):
    # Predict the brain tumor
            predicted_class, class_probabilities = predict_brain_tumor(uploaded_file)
            if predicted_class == 'breast_bengin':
                st.info("The tumor is benign")
                st.success("Benign breast conditions are non-cancerous changes in breast tissue that can cause symptoms similar to breast cancer. It's important to note that while these conditions may cause discomfort or concern, they are not cancerous. However, some benign breast conditions may increase the risk of developing breast cancer in the future. Here are some common benign breast conditions and their associated symptoms:")
            elif predicted_class == 'breast_malignant':
                st.info("The tumor is malignant")
                st.success("Breast cancer is a type of cancer that starts in the breast. Cancer starts when cells begin to grow out of control. Breast cancer cells usually form a tumor that can often be seen on an x-ray or felt as a lump.")
