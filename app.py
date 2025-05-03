import streamlit as st # Imports the Streamlit library, which is used to create interactive web applications.
from PIL import Image # Imports the Image module from the Pillow library, used for opening and manipulating image files.
import cv2 # Imports the OpenCV library, which provides tools for computer vision tasks like image processing.
import numpy as np # Imports the NumPy library, which is fundamental for numerical operations in Python, especially with arrays.

st.title("Insurance Card OCR") # Sets the title of the Streamlit web application that will be displayed in the browser.

uploaded_file = st.file_uploader("Upload an insurance card image", type=["jpg", "jpeg", "png"]) # Creates a file uploader widget in the Streamlit app, allowing users to upload image files with the specified types.

if uploaded_file is not None: # Checks if a file has been uploaded by the user.
    image = Image.open(uploaded_file) # Opens the uploaded image file using the Pillow library.
    st.image(image, caption="Uploaded Image", use_column_width=True) # Displays the uploaded image in the Streamlit app, with a caption and adjusted to the width of the column.
    st.subheader("OCR Result:") # Displays a subheader in the Streamlit app to introduce the OCR result section.
    # Add OCR processing code here

    if st.button("Process with EasyOCR"): # Creates a button in the Streamlit app labeled "Process with EasyOCR". The code inside this block will execute when the button is clicked.
        image_np = np.array(image) # Converts the Pillow image object into a NumPy array, which is often required by image processing libraries like EasyOCR and OpenCV.
        import easyocr # Imports the EasyOCR library, which is used for Optical Character Recognition.
        reader = easyocr.Reader(['en']) # Initializes the EasyOCR reader object, specifying that it should recognize English text.
        results = reader.readtext(image_np, detail=0) # Performs OCR on the input NumPy array representing the image. The 'detail=0' argument tells EasyOCR to return only the recognized text, without bounding box information or confidence scores.
        markdown_output = "" # Initializes an empty string variable to store the OCR results in Markdown format.
        for text in results: # Iterates through the list of recognized text strings returned by EasyOCR.
            markdown_output += f"- {text}\n" # Appends each recognized text string to the markdown_output string, formatting it as a bullet point in Markdown.
        st.markdown(markdown_output) # Displays the formatted OCR result in the Streamlit app using Markdown.

    if st.button("Process with PaddleOCR"): # Creates a button in the Streamlit app labeled "Process with PaddleOCR". The code inside this block will execute when the button is clicked.
        image_np = np.array(image) # Converts the Pillow image object into a NumPy array.
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # Converts the NumPy array representing the image from RGB color space (used by Pillow) to BGR color space (used by OpenCV).
        from paddleocr import PaddleOCR # Imports the PaddleOCR library for Optical Character Recognition.
        ocr = PaddleOCR(use_angle_cls=True, lang='en') # Initializes the PaddleOCR object, enabling angle classification (to handle rotated text) and specifying English as the language for recognition.
        results = ocr.ocr(image_cv, cls=True) # Performs OCR on the input OpenCV image. The 'cls=True' argument enables text angle classification.
        markdown_output = "" # Initializes an empty string to store the OCR results in Markdown format.
        for line in results: # Iterates through the list of lines of text detected by PaddleOCR.
            for word_info in line: # Iterates through the information about each word within a detected line.
                if isinstance(word_info, list) and len(word_info) > 1: # Checks if the word information is a list and contains more than one element, which typically indicates the bounding box and the text.
                    text = word_info[1] # Extracts the recognized text, which is usually the second element in the word information list.
                    markdown_output += f"- {text}\n" # Appends the extracted text to the markdown_output string, formatted as a Markdown bullet point.
        st.markdown(markdown_output) # Displays the formatted OCR result in the Streamlit app using Markdown.

def preprocess_image(image): # Defines a function named 'preprocess_image' that takes an image object as input.
    img_array = np.array(image) # Converts the input Pillow image object into a NumPy array.
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) # Converts the color image (NumPy array) to grayscale using OpenCV.
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] # Applies Otsu's thresholding to the grayscale image. This method automatically determines the optimal threshold value to separate the foreground (text) from the background. cv2.THRESH_BINARY_INV inverts the result so that the text is white on a black background. [1] at the end extracts the thresholded image.
    return Image.fromarray(thresh) # Converts the processed NumPy array back into a Pillow image object and returns it.

if uploaded_file is not None: # Checks again if a file has been uploaded.
    #... (image loading)...
    if st.checkbox("Apply Image Preprocessing"): # Creates a checkbox in the Streamlit app labeled "Apply Image Preprocessing".
        processed_image = preprocess_image(image) # If the checkbox is checked, it calls the 'preprocess_image' function to process the uploaded image.
        st.image(processed_image, caption="Preprocessed Image", use_column_width=True) # Displays the preprocessed image in the Streamlit app.
        # Use processed_image for OCR
        image_to_ocr = processed_image # Sets the 'image_to_ocr' variable to the preprocessed image, so this image will be used for OCR if preprocessing is enabled.
    else: # If the "Apply Image Preprocessing" checkbox is not checked.
        # Use original image for OCR
        image_to_ocr = image # Sets the 'image_to_ocr' variable to the original uploaded image.
        pass
    #... (OCR processing using EasyOCR or PaddleOCR)...
