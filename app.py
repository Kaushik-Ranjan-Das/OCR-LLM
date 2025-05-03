import streamlit as st
from PIL import Image
import numpy as np
import cv2
import requests
import base64
import os
import io
import json

# Set page configuration
st.set_page_config(
    page_title="Insurance Card OCR",
    page_icon="📷",
    layout="wide"
)

st.title("Insurance Card OCR")

# Add a sidebar with information
with st.sidebar:
    st.info("Upload an insurance card image to extract text using OCR.")
    st.subheader("OCR Model Selection")
    ocr_method = st.radio(
        "Select OCR Method",
        ["Llama 3.2 Vision", "EasyOCR", "PaddleOCR"]
    )
    
    if ocr_method == "Llama 3.2 Vision":
        llama_model = st.selectbox(
            "Select Llama Model",
            ["Llama-3.2-90B-Vision", "Llama-3.2-11B-Vision", "free"]
        )
        
        # API key input (could use st.secrets in production)
        together_api_key = st.text_input("Together AI API Key", type="password")
        if not together_api_key:
            st.warning("Please enter your Together AI API key")
            if 'TOGETHER_API_KEY' in os.environ:
                together_api_key = os.environ['TOGETHER_API_KEY']
                st.success("Using API key from environment variable")

# Define preprocessing function
def preprocess_image(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return Image.fromarray(thresh)

# Function to process image with Llama Vision model
def process_with_llama_vision(image, api_key, model="Llama-3.2-90B-Vision"):
    # Convert image to base64
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Determine which model to use
    vision_llm = f"meta-llama/{model}-Instruct-Turbo" if model != "free" else "meta-llama/Llama-Vision-Free"
    
    # System prompt for OCR
    system_prompt = """
    Extract and organize all text from this insurance card image. Identify and label key information such as:
    - Member name
    - Member ID number
    - Group number
    - Plan type
    - Issuer/Insurance company
    - Contact information
    - Copay/deductible information
    - Prescription information
    - Any other relevant details
    
    Format the information clearly and organize it logically. If any field is unclear or not present, indicate that.
    """
    
    # Prepare API request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": vision_llm,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                ]
            }
        ]
    }
    
    # Make API request
    try:
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()  # Raise exception for HTTP errors
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error processing with Llama Vision: {str(e)}"

# Create the file uploader
uploaded_file = st.file_uploader("Upload an insurance card image", type=["jpg", "jpeg", "png"])

# Main app logic
if uploaded_file is not None:
    try:
        # Load and display the image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Preprocessing option
        apply_preprocessing = st.checkbox("Apply Image Preprocessing")
        
        if apply_preprocessing:
            processed_image = preprocess_image(image)
            with col2:
                st.image(processed_image, caption="Preprocessed Image", use_column_width=True)
            image_to_ocr = processed_image
        else:
            image_to_ocr = image
        
        st.subheader("OCR Result:")
        
        # Process with selected OCR method
        if ocr_method == "Llama 3.2 Vision":
            if st.button("Process with Llama Vision"):
                if not together_api_key:
                    st.error("Please provide a Together AI API key in the sidebar")
                else:
                    with st.spinner("Processing with Llama Vision..."):
                        result = process_with_llama_vision(image_to_ocr, together_api_key, llama_model)
                        st.markdown(result)
        
        elif ocr_method == "EasyOCR":
            if st.button("Process with EasyOCR"):
                with st.spinner("Processing with EasyOCR..."):
                    try:
                        image_np = np.array(image_to_ocr)
                        import easyocr
                        reader = easyocr.Reader(['en'])
                        results = reader.readtext(image_np, detail=0)
                        
                        if results:
                            markdown_output = ""
                            for text in results:
                                markdown_output += f"- {text}\n"
                            st.markdown(markdown_output)
                        else:
                            st.warning("No text detected. Try adjusting the image or using a different OCR method.")
                    except Exception as e:
                        st.error(f"Error during EasyOCR processing: {e}")
        
        elif ocr_method == "PaddleOCR":
            if st.button("Process with PaddleOCR"):
                with st.spinner("Processing with PaddleOCR..."):
                    try:
                        image_np = np.array(image_to_ocr)
                        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                        from paddleocr import PaddleOCR
                        ocr = PaddleOCR(use_angle_cls=True, lang='en')
                        results = ocr.ocr(image_cv, cls=True)
                        
                        if results and any(results):
                            markdown_output = ""
                            for line in results:
                                if line:  # Check if line is not empty
                                    for word_info in line:
                                        if isinstance(word_info, list) and len(word_info) > 1:
                                            text = word_info[1][0]  # Updated to handle PaddleOCR's output structure
                                            confidence = word_info[1][1]
                                            markdown_output += f"- {text} (Confidence: {confidence:.2f})\n"
                            st.markdown(markdown_output)
                        else:
                            st.warning("No text detected. Try adjusting the image or using a different OCR method.")
                    except Exception as e:
                        st.error(f"Error during PaddleOCR processing: {e}")
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please ensure you've uploaded a valid image file.")

else:
    # Display sample image or instructions when no file is uploaded
    st.info("👆 Upload an insurance card image to get started!")
