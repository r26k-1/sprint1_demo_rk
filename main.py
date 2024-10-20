from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pymongo import MongoClient
import pandas as pd
import spacy
import fitz  # PyMuPDF for PDF handling
from docx import Document  # For DOCX file handling
from utils import preprocess_dataframe, get_sentiment, generate_summary
from fastapi.staticfiles import StaticFiles
import re
import io
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# MongoDB client setup
client = MongoClient("mongodb://localhost:27017/")
db = client['file_data']
collection = db['documents']

# Serve HTML templates
templates = Jinja2Templates(directory="templates")

# Load SpaCy model for preprocessing
nlp = spacy.load("en_core_web_sm")

# Function to preprocess text data
def preprocess_text(text: str) -> str:
    """Clean and preprocess the text data."""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    cleaned_text = ' '.join(tokens)
    return cleaned_text

def read_pdf(file: UploadFile) -> str:
    """Extract text from a PDF file."""
    pdf_reader = fitz.open(stream=file.file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_reader)):
        page = pdf_reader.load_page(page_num)
        text += page.get_text()
    return text

# DOCX reader function
def read_docx(file: UploadFile) -> str:
    """Extract text from a DOCX file."""
    doc = Document(file.file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def generate_csv_summary(df):
    """Generate a generic summary for a DataFrame."""
    summary_lines = []
    total_records = len(df)
    summary_lines.append(f"The dataset includes {total_records} records.")
    attributes = df.columns.tolist()
    summary_lines.append("Key attributes include: " + ", ".join(attributes) + ".")
    
    
    
    return "\n".join(summary_lines)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Homepage or root path."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload/")
async def upload_file(request: Request, file: UploadFile = File(...)):
    """Upload a file and store its contents in MongoDB."""
    try:
        df = None
        raw_text = None
        summary = ''
        sentiment = ''

        # Store the file in MongoDB
        document = {
            "filename": file.filename,
            "content": await file.read(),  # Read the file content once
        }

        # Insert the document into MongoDB and get the inserted ID
        result = collection.insert_one(document)
        file_id = str(result.inserted_id)

        # Handle CSV and Excel files (convert to DataFrame)
        if file.filename.endswith('.csv'):
            # Move to the start of the stream after reading
            await file.seek(0)
            df = pd.read_csv(file.file)
        elif file.filename.endswith('.xlsx'):
            await file.seek(0)
            df = pd.read_excel(file.file)
        else:
            # Handle TXT, PDF, DOCX files (only extract text, no DataFrame)
            await file.seek(0)  # Ensure the stream is at the beginning
            if file.filename.endswith('.txt'):
                raw_text = document["content"].decode('utf-8')
            elif file.filename.endswith('.pdf'):
                raw_text = read_pdf(file)
            elif file.filename.endswith('.docx'):
                raw_text = read_docx(file)
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")

        # Generate summary and sentiment analysis for text-based files (TXT, PDF, DOCX)
        if raw_text:
            summary = generate_summary(raw_text)
            sentiment = get_sentiment(raw_text)  # Perform sentiment analysis

            # Store text and summary in MongoDB
            collection.insert_one({"filename": file.filename, "text": raw_text, "summary": summary})

            return templates.TemplateResponse("results.html", {
                "request": request,
                "raw_text": raw_text,
                "summary": summary,
                "sentiment": sentiment,
                "file_type": "text"
            })

        # Process DataFrame for CSV and Excel files
        if df is not None:
            # Preprocess the DataFrame
            df = preprocess_dataframe(df)

            # Generate summary for the DataFrame
            summary = generate_csv_summary(df)
            # Capture DataFrame info as a string
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue() 
            # Insert into MongoDB
            records = df.to_dict(orient='records')
            if records:
                collection.insert_many(records)

            return templates.TemplateResponse("results.html", {
                "request": request,
                "dataframe": df.to_html(classes='dataframe'),
                "summary": summary,
                "sentiment": sentiment,
                "file_type": "dataframe",
                "info": info_str,  # Include DataFrame info
                "description": df.describe() if df is not None else None  # Include DataFrame description
            })

        return templates.TemplateResponse("index.html", {"request": request, "file_id": file_id})

    except Exception as e:
        print(f"Error: {e}")  # Log the error to the console
        raise HTTPException(status_code=500, detail=str(e))
