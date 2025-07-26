import streamlit as st
from PIL import Image
import requests
import io

st.set_page_config(page_title="Breast Cancer Prediction", layout="centered")
st.title("🧠 Breast Cancer Prediction with GradCAM")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

# Check if an image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("Classifying...")

    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Send POST request to Flask API
    files = {'image': ('image.png', img_byte_arr, 'image/png')}
    try:
        response = requests.post("http://127.0.0.1:1234/cam", files=files)
        if response.status_code == 200:
            data = response.json()
            st.success("Prediction successful!")
            st.markdown(f"**Predicted Class:** `{data['predicted_class']}`")
            st.markdown(f"**Probability:** `{data['probability']:.4f}`")

            # Display Grad-CAM result
            grad_cam_url = f"http://127.0.0.1:1234{data['image_url']}"
            st.image(grad_cam_url, caption='Grad-CAM Output', use_container_width=True)
        else:
            st.error("Error from server:")
            st.text(response.text)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not connect to the Flask API.\nDetails: {e}")
