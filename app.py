import os
import streamlit as st
import pytesseract
from PIL import Image
import io
import pdfplumber
from groq import Groq

# Step 1: Set up the API key for Groq
api_key = st.secrets["api_key"]

# Configure pytesseract to use the installed Tesseract executable
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Adjust the path as necessary

# Step 2: Function to extract text from images within a PDF
def extract_text_from_pdf(pdf_file):
    extracted_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            # Try to extract text directly
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text
            else:
                # If no text found, convert the page to an image and use OCR
                image = page.to_image(resolution=300)
                pil_image = image.original.convert("RGB")
                extracted_text += pytesseract.image_to_string(pil_image)
    return extracted_text

# Step 3: Initialize Groq API Client
client = Groq(api_key=api_key)

# Step 4: Function to query the LLM
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

# Step 5: Streamlit GUI
st.title("OCR-based PDF AI Chatbot")

# Step 6: Upload PDF
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

# Step 7: Summary and Q&A functionality
if pdf_file:
    st.write("PDF uploaded successfully!")

    # Button to extract text from PDF
    if st.button("Extract Text"):
        extracted_text = extract_text_from_pdf(pdf_file)
        if extracted_text:
            st.subheader("Extracted Text from PDF")
            st.write(extracted_text)
        else:
            st.write("No text found in the PDF.")

    # Text input to ask a question
    user_query = st.text_input("Ask a question about the PDF content:")

    if user_query and 'extracted_text' in locals() and extracted_text:
        # Query the LLM with the extracted text and user question
        response = query_llm(user_query, extracted_text)
        st.subheader("AI Response")
        st.write(response)
