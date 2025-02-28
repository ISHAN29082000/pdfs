from fastapi import FastAPI, UploadFile, File, Query
import os
import shutil
from pypdf import PdfReader
from transformers import pipeline

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load AI model (Hugging Face's RoBERTa)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Store extracted text from PDFs
pdf_text_data = {}

@app.post("/upload/")
async def upload_files(files: list[UploadFile] = File(...)):
    """Handles multiple PDF uploads and extracts text."""
    global pdf_text_data
    pdf_text_data.clear()  # Clear previous data

    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text from PDF
        reader = PdfReader(file_path)
        extracted_text = ""
        for page in reader.pages:
            extracted_text += page.extract_text() + "\n"

        pdf_text_data[file.filename] = extracted_text

    return {"message": "PDFs uploaded and processed!"}

@app.get("/chat/")
async def chat(query: str = Query(..., description="Your question")):
    """Processes user query and fetches relevant answers."""
    if not pdf_text_data:
        return {"response": "No PDFs uploaded yet!"}

    combined_text = " ".join(pdf_text_data.values())

    response = qa_pipeline({
        "question": query,
        "context": combined_text
    })

    return {"response": response["answer"]}

