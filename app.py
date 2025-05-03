import streamlit as st
from PIL import Image
import numpy as np
import cv2
import requests
import base64
import os
import io
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
TOGETHER_API_KEY: "5eae8fec1b3c9df67e61157cf2c6808ae8f4d68a8a8d0747a9da6d66bbc681c7"

# Set page configuration
st.set_page_config(
    page_title="Insurance Card OCR",
    page_icon=":camera:",
    layout="wide"
)

st.title("Insurance Card OCR")

# Add a sidebar with information
with st.sidebar:
    st.info("Upload an insurance card image to extract text using OCR.")
    
    st.subheader("OCR Model Selection")
    
    # Highlighted recommendation box for Llama model
    st.write("✨ **Recommended Method** ✨")
    st.success("""
    **Llama 3.2 Vision** provides the most accurate insurance card text extraction with structured output.
    """)
    
    ocr_method = st.radio(
        "Select OCR Method",
        ["Llama 3.2 Vision", "EasyOCR", "PaddleOCR"],
        index=0  # Default to Llama 3.2 Vision
    )
    
    if ocr_method == "Llama 3.2 Vision":
        llama_model = st.selectbox(
            "Select Llama Model",
            ["Llama-3.2-90B-Vision", "Llama-3.2-11B-Vision", "free"],
            index=0  # Default to 90B
        )
        
        # Check for API key in environment variables only
        together_api_key = os.environ.get('TOGETHER_API_KEY')
        if not together_api_key:
            st.warning("TOGETHER_API_KEY not found in environment variables. Please set it before running the app.")
            st.code("export TOGETHER_API_KEY='your_api_key_here'")
        else:
            st.success("✓ Together AI API key found in environment variables")

# Define preprocessing function
def preprocess_image(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return Image.fromarray(thresh)

# Function to process image with Llama Vision model
def process_with_llama_vision(image, api_key, model="Llama-3.2-90B-Vision"):
    try:
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Determine which model to use
        vision_llm = f"meta-llama/{model}-Instruct-Turbo" if model != "free" else "meta-llama/Llama-Vision-Free"
        
        # System prompt for OCR - structured for table output
        system_prompt = """
        Extract all text from this insurance card image and organize it into a structured format.
        
        Format your response as a clear table with two columns:
        1. Field name (e.g., "Insurance Company", "Member ID", etc.)
        2. Value extracted from the card
        
        Include all relevant fields such as:
        - Insurance Company/Plan Name
        - Member Name
        - Member ID/Identification Number
        - Group Number
        - Plan Type/Plan ID
        - Effective Date
        - Expiration Date
        - PCP Copay
        - Specialist Copay
        - Emergency Room Copay
        - Prescription Information
        - Customer Service Phone Numbers
        - Claims Address
        - Website
        
        For any field where information is not visible or not present on the card, indicate "Not found" as the value.
        Present the information in markdown table format for clear readability.
        """
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Create payload 
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
        
        # Convert payload to JSON with ASCII encoding
        json_payload = json.dumps(payload, ensure_ascii=True)
        
        # Make API request
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            data=json_payload
        )
        response.raise_for_status()  # Raise exception for HTTP errors
        
        result = response.json()
        response_content = result["choices"][0]["message"]["content"]
        
        # Make sure response has a table format, add if missing
        if "| Field | Value |" not in response_content and "| --- | --- |" not in response_content:
            # Parse the content and convert to table if needed
            lines = response_content.strip().split('\n')
            table_content = "| Field | Value |\n| --- | --- |\n"
            
            current_field = ""
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if line has a field:value pattern
                if ":" in line:
                    parts = line.split(":", 1)
                    field = parts[0].strip()
                    value = parts[1].strip() if len(parts) > 1 else "Not found"
                    table_content += f"| {field} | {value} |\n"
                else:
                    # Add as additional information
                    table_content += f"| Additional Info | {line} |\n"
            
            return table_content
        
        return response_content
    except Exception as e:
        st.error(f"Full error: {str(e)}")
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
            if st.button("Process with Llama Vision", type="primary"):
                if not together_api_key:
                    st.error("TOGETHER_API_KEY not found in environment variables. Please set it before running the app.")
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
                            # Format results as a table
                            table_content = "| Text Detected |\n| --- |\n"
                            for text in results:
                                table_content += f"| {text} |\n"
                            st.markdown(table_content)
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
                            # Format results as a table
                            table_content = "| Text Detected | Confidence |\n| --- | --- |\n"
                            for line in results:
                                if line:  # Check if line is not empty
                                    for word_info in line:
                                        if isinstance(word_info, list) and len(word_info) > 1:
                                            text = word_info[1][0]  # Updated to handle PaddleOCR's output structure
                                            confidence = word_info[1][1]
                                            table_content += f"| {text} | {confidence:.2f} |\n"
                            st.markdown(table_content)
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
