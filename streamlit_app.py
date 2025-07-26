import streamlit as st
from PIL import Image
import requests
import io

st.set_page_config(page_title="Breast Cancer Detection", layout="centered")

st.title("🩺 Breast Cancer Prediction with GradCAM")

# Upload image
uploaded_file = st.file_uploader("📤 Upload an image...", type=["jpg", "png", "jpeg"])

# Clear previous output when new image is uploaded
if uploaded_file is not None:
    # Open and display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)
    st.write("🔍 Processing...")

    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Send request to Flask API
    files = {'image': ('image.png', img_byte_arr, 'image/png')}
    try:
        response = requests.post("http://127.0.0.1:1234/cam", files=files)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.error(f"🚨 Error communicating with backend: {e}")
    else:
        data = response.json()
        st.success("✅ Prediction Complete!")

        st.markdown(f"**🔬 Predicted Class:** `{data['predicted_class']}`")
        st.markdown(f"**📊 Probability:** `{data['probability']:.4f}`")

        # Show GradCAM result image
        grad_cam_url = f"http://127.0.0.1:1234{data['image_url']}"
        st.image(grad_cam_url, caption='🔥 Grad-CAM Visualization', use_column_width=True)
