import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
import io
from groq import Groq
import os

# Step 1: Set up API key for Groq
api_key = st.secrets["api_key"]

# Step 2: Function to extract text from PDF with OCR fallback
def extract_text_from_pdf(pdf_file):
    extracted_text = ""
    
    # Try extracting text using pdfplumber
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # If text is found, add it to the extracted_text
                extracted_text += page_text
            else:  # If no text found, use OCR
                # Convert PDF page to an image for OCR
                img = page.to_image()
                pil_image = img.original  # Get the PIL image
                
                # Use Tesseract OCR to extract text from the image
                text = pytesseract.image_to_string(pil_image)
                extracted_text += text

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
st.title("PDF AI Chatbot with Summary and Q&A")

# Step 6: Upload PDF
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

# Initialize extracted_text
extracted_text = ""

# Step 7: Summary and Q&A functionality
if pdf_file:
    st.write("PDF uploaded successfully!")
    
    # Button to extract and summarize text
    if st.button("Summarize PDF"):
        extracted_text = extract_text_from_pdf(pdf_file)
        if extracted_text.strip():  # Check if any text was extracted
            st.subheader("Extracted Text from PDF")
            st.write(extracted_text)
        else:
            st.warning("No text found in the PDF.")

    # Text input to ask a question
    user_query = st.text_input("Ask a question about the PDF content:")
    
    if user_query and extracted_text:
        # Query the LLM with the extracted text and user question
        response = query_llm(user_query, extracted_text)
        st.subheader("AI Response")
        st.write(response)
    elif user_query:  # If user has entered a query but no text is extracted
        st.warning("Please summarize the PDF first before asking questions.")
