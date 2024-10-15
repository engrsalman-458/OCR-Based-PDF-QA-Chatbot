import streamlit as st
import pdfplumber
from groq import Groq

# Step 1: Set up API key for Groq
import os

api_key = st.secrets["api_key"]


# Step 2: Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    extracted_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted_text += page.extract_text()
    return extracted_text

# Step 3: Initialize Groq API Client
client = Groq(api_key=os.environ.get("api_key"))

# Step 4: Function to query the LLM
def query_llm(prompt, context):
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

# Step 7: Summary and Q&A functionality
if pdf_file:
    st.write("PDF uploaded successfully!")
    
    # Button to extract and summarize text
    if st.button("Summarize PDF"):
        extracted_text = extract_text_from_pdf(pdf_file)
        st.subheader("Extracted Text from PDF")
        st.write(extracted_text)
    
    # Text input to ask a question
    user_query = st.text_input("Ask a question about the PDF content:")
    
    if user_query:
        # Query the LLM with the extracted text and user question
        response = query_llm(user_query, extracted_text)
        st.subheader("AI Response")
        st.write(response)

