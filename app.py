import streamlit as st
import fitz  # PyMuPDF for PDF handling
import pytesseract
from PIL import Image
import io
from groq import Groq
import os

# Step 1: Set up API key for Groq
api_key = st.secrets["api_key"]  # Make sure to set your API key in Streamlit secrets

# Step 2: Configure pytesseract for OCR
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Step 3: Function to extract text from images within a PDF
def extract_text_from_pdf_images(pdf_file):
    extracted_text = ""
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    # Loop through each page in the PDF
    for page_num in range(doc.page_count):
        page = doc[page_num]
        images = page.get_images(full=True)

        # Extract images from the page and run OCR
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_data = base_image["image"]

            # Convert the image data to a PIL image
            image = Image.open(io.BytesIO(image_data))

            # Use Tesseract OCR to extract text from the image
            text = pytesseract.image_to_string(image)
            extracted_text += f"Page {page_num + 1}, Image {img_index + 1}:\n{text}\n"

    return extracted_text

# Step 4: Initialize Groq API Client
client = Groq(api_key=api_key)

# Step 5: Function to query the LLM
def query_llm(prompt, context):
    """Query the LLM with the extracted text and user question."""
    messages = [
        {"role": "system", "content": f"The context is: {context}"},
        {"role": "user", "content": prompt}
    ]
    chat_completion = client.chat.completions.create(
        messages=messages,
        model="llama3-8b-8192"
    )
    return chat_completion.choices[0].message.content

# Step 6: Streamlit GUI
st.title("OCR Based PDF AI Chatbot and Q&A")

# Step 7: Upload PDF
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

# Step 8: Automatically extract text when a PDF is uploaded
if pdf_file:
    st.write("PDF uploaded successfully!")
    
    # Extract text from the PDF immediately
     extracted_text = extract_text_from_pdf_images(pdf_file)
     st.subheader("Extracted Text from PDF Images")
    # st.write(extracted_text)

    # Text input to ask a question
    user_query = st.text_input("Ask a question about the PDF content:")
    
    if user_query and extracted_text:
        # Query the LLM with the extracted text and user question
        response = query_llm(user_query, extracted_text)
        st.subheader("AI Response")
        st.write(response)
