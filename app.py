import streamlit as st
import fitz  # PyMuPDF for PDF handling
import pytesseract
from PIL import Image
import io
from sentence_transformers import SentenceTransformer
from groq import Groq
import os

# Step 1: Set up API key for Groq
api_key = st.secrets["api_key"]  # Ensure the API key is set in Streamlit secrets

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

# Step 4: Initialize Sentence Transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 5: Initialize Groq API Client
client = Groq(api_key=api_key)

# Step 6: Function to chunk the extracted text
def chunk_text(text, chunk_size=1000):
    """Splits the text into chunks of specified size."""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Step 7: Generate embeddings for each chunk
def generate_embeddings(text_chunks):
    """Generates embeddings for a list of text chunks."""
    embeddings = model.encode(text_chunks, convert_to_tensor=True)
    return embeddings

# Step 8: Function to convert user query to an embedding
def get_query_embedding(query):
    """Generates an embedding for the user's query."""
    return model.encode(query, convert_to_tensor=True)

# Step 9: Function to find the most relevant chunks using cosine similarity
from sentence_transformers import util

def find_relevant_chunks(query_embedding, pdf_embeddings, pdf_chunks, top_n=3):
    """Finds the top N most relevant text chunks using cosine similarity."""
    similarities = util.pytorch_cos_sim(query_embedding, pdf_embeddings)
    top_results = similarities.argsort(descending=True)[:top_n]
    relevant_chunks = [pdf_chunks[i] for i in top_results]
    return relevant_chunks

# Step 10: Function to query the LLM using relevant chunks
def query_llm_with_relevant_chunks(prompt, relevant_chunks):
    """Query the LLM using the most relevant chunks of text."""
    context = " ".join(relevant_chunks)
    return query_llm(prompt, context)

# Step 11: Query the LLM
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

# Step 12: Streamlit GUI
st.title("OCR Based PDF AI Chatbot and Q&A")

# Step 13: Upload PDF
pdf_file = st.file_uploader("Upload a PDF", type="pdf")

# Step 14: Automatically extract text when a PDF is uploaded but don't display it
if pdf_file:
    st.write("PDF uploaded successfully!")
    
    # Extract text from the PDF immediately
    extracted_text = extract_text_from_pdf_images(pdf_file)

    # Split text into chunks and generate embeddings
    pdf_chunks = chunk_text(extracted_text)
    pdf_embeddings = generate_embeddings(pdf_chunks)

    # Text input to ask a question
    user_query = st.text_input("Ask a question about the PDF content:")
    
    if user_query:
        # Convert the query to an embedding
        query_embedding = get_query_embedding(user_query)
        
        # Find the most relevant chunks of the PDF
        relevant_chunks = find_relevant_chunks(query_embedding, pdf_embeddings, pdf_chunks)
        
        # Query the LLM with the relevant chunks
        response = query_llm_with_relevant_chunks(user_query, relevant_chunks)
        st.subheader("AI Response")
        st.write(response)

    # Option to display the extracted text if the user wants
    if st.button("Show Extracted Text"):
        st.subheader("Extracted Text from PDF Images")
        st.write(extracted_text)
