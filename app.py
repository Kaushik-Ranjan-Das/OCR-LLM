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
# Correctly set the API key
TOGETHER_API_KEY = "5eae8fec1b3c9df67e61157cf2c6808ae8f4d68a8a8d0747a9da6d66bbc681c7"

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
    
    st.subheader("OCR Selection")
    
    # Highlighted recommendation box for Llama model
    st.write("✨ **Recommended Method** ✨")
    st.success("""
    **Note that we are using an open source model and have limited free access API key. Please contact kaushikranjan@gmail.com if the app does not work for you
    """)
   
    # Fixed: Use the correct model identifier
    llama_model = "Llama-3.1-8B-Vision"  # Changed to a model that's likely available
          
# Use the hardcoded API key if environment variable isn't available
together_api_key = os.environ.get('TOGETHER_API_KEY', TOGETHER_API_KEY)

# Define preprocessing function
def preprocess_image(image):
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return Image.fromarray(thresh)

# Function to process image with Llama Vision model
def process_with_llama_vision(image, api_key, model):
    try:
        # Convert image to base64
        buffer = io.BytesIO()
        # Convert RGBA to RGB if needed
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(buffer, format="JPEG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Set the model correctly - meta-llama or different prefix may be needed
        # Try different model identifiers if this one doesn't work
        vision_llm = model
        
        # Debug model name
        st.write(f"Using model: {vision_llm}")
        
        # System prompt for OCR - structured for table output
        user_prompt = """
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
        
        # Create payload - adjust based on Together.ai's API
        payload = {
            "model": vision_llm,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                    ]
                }
            ],
            "temperature": 0.2,  # Lower temperature for more deterministic output
            "max_tokens": 1024    # Ensure enough tokens for the response
        }
        
        # Convert payload to JSON with ASCII encoding
        json_payload = json.dumps(payload, ensure_ascii=True)
        
        # Show the request payload for debugging (without the image data)
        debug_payload = payload.copy()
        if "messages" in debug_payload and len(debug_payload["messages"]) > 0:
            for msg in debug_payload["messages"]:
                if "content" in msg and isinstance(msg["content"], list):
                    for i, content_item in enumerate(msg["content"]):
                        if isinstance(content_item, dict) and "image_url" in content_item:
                            msg["content"][i]["image_url"]["url"] = "[BASE64_IMAGE_DATA]"
        
        st.write("Request payload structure:")
        st.json(debug_payload)
        
        # Make API request
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers=headers,
            data=json_payload
        )
        
        # Add debug information
        st.write(f"API Status Code: {response.status_code}")
        
        # Check for error response and display details
        if response.status_code != 200:
            st.error(f"API Error: {response.status_code}")
            st.error(f"Response: {response.text}")
            return f"Error: API returned status code {response.status_code}. Response: {response.text}"
        
        # Parse the response
        result = response.json()
        st.write("API Response structure:")
        
        # Create a safe copy of the response for display
        safe_result = result.copy()
        st.json(safe_result)
        
        # Extract the content from the response
        response_content = result["choices"][0]["message"]["content"]
        
        # Make sure response has a table format, add if missing
        if "| Field | Value |" not in response_content and "| --- | --- |" not in response_content:
            # Parse the content and convert to table if needed
            lines = response_content.strip().split('\n')
            table_content = "| Field | Value |\n| --- | --- |\n"
            
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

# Function to list available models from Together API
def list_available_models(api_key):
    try:
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        
        response = requests.get(
            "https://api.together.xyz/v1/models",
            headers=headers
        )
        
        if response.status_code == 200:
            models = response.json()
            vision_models = [m for m in models["data"] if "vision" in m["id"].lower() or "llava" in m["id"].lower()]
            return vision_models
        else:
            return []
    except Exception as e:
        st.error(f"Error listing models: {str(e)}")
        return []

# Create the file uploader
uploaded_file = st.file_uploader("Upload an insurance card image", type=["jpg", "jpeg", "png"])

# Show available models in sidebar
with st.sidebar:
    st.subheader("Available Models")
    if st.button("Check Available Vision Models"):
        vision_models = list_available_models(together_api_key)
        if vision_models:
            st.write("Available vision models:")
            for model in vision_models:
                st.write(f"- {model['id']}")
        else:
            st.write("Could not retrieve models or no vision models found.")

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
        
        # Allow custom model input
        custom_model = st.text_input("Enter model ID (leave empty to use default)", 
                                    value=llama_model)
        if custom_model:
            model_to_use = custom_model
        else:
            model_to_use = llama_model
            
        # Process with Llama Vision
        if st.button("Process with Vision Model", type="primary"):
            if not together_api_key:
                st.error("TOGETHER_API_KEY not found in environment variables. Please set it before running the app.")
            else:
                with st.spinner(f"Processing with model {model_to_use}..."):
                    result = process_with_llama_vision(image_to_ocr, together_api_key, model_to_use)
                    st.markdown(result)
                    
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please ensure you've uploaded a valid image file.")

else:
    # Display sample image or instructions when no file is uploaded
    st.info("👆 Upload an insurance card image to get started!")
