# app.py bryant part
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import openai
from dotenv import load_dotenv
import logging
import traceback
import html
import re

# Import modules for text extraction, data chunking, and conversation history
from TextExtraction import get_pdf_text, get_docx_text, get_pptx_text, get_xlsx_text
from DataChunking import load_text, split_text, split_list, create_db
from modelWithConvoHist import get_convo_hist_answer

# Logging config
logging.basicConfig(
    level=logging.INFO,
    filename='app.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Load environment variables from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
UPLOAD_FOLDER = 'uploads'
DATA_PATH = os.getenv("DATA_PATH", "./modelling/extracted_text.txt")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'xlsx', 'pptx'}

# Ensure 'uploads' folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

logging.info("Flask application initialized. Upload folder set up at %s", UPLOAD_FOLDER)

# Helper function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to extract text based on file type
def extract_text(filepath, file_type):
    try:
        if file_type == 'pdf':
            return get_pdf_text([filepath])
        elif file_type == 'docx':
            return get_docx_text([filepath])
        elif file_type == 'xlsx':
            return get_xlsx_text([filepath])
        elif file_type == 'pptx':
            return get_pptx_text([filepath])
    except Exception as e:
        logging.error("Error extracting text from %s: %s", filepath, str(e))
        raise
    return ""

# Helper function to sanitize user input
def sanitize_input(input_text):
    sanitized = html.escape(input_text)
    sanitized = re.sub(r"[^a-zA-Z0-9\s\.,!?'-]", "", sanitized)
    if len(sanitized) > 500:
        sanitized = sanitized[:500]
    return sanitized

# Route for file uploads
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            logging.warning("No file part in the request.")
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '' or not allowed_file(file.filename):
            logging.warning("Invalid file type or empty filename: %s", file.filename)
            return jsonify({'error': 'Invalid file type'}), 400

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info("File saved: %s", filepath)

        # Extract and save text to DATA_PATH
        file_type = filename.rsplit('.', 1)[1].lower()
        text = extract_text(filepath, file_type)
        with open(DATA_PATH, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        logging.info("Text extracted and saved to %s", DATA_PATH)

        # Process extracted text for chunking and updating vector database
        documents = load_text(DATA_PATH)
        chunks = split_text(documents)
        split_chunked = list(split_list(chunks, 166))
        create_db(split_chunked)
        logging.info("Vector database updated.")

        return jsonify({'status': 'File uploaded and text extracted'}), 200
    except Exception as e:
        logging.error("Error in /upload endpoint: %s", traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

# Route for question answering with conversation history
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        question = request.json.get('question')
        if not question:
            logging.warning("No question provided in request.")
            return jsonify({'error': 'No question provided'}), 400

        # Sanitize user input
        sanitized_question = sanitize_input(question)
        logging.info("Sanitized question: %s", sanitized_question)

        # Get answer from model with conversation history
        response = get_convo_hist_answer(sanitized_question)
        logging.info("Question answered: %s", sanitized_question)
        return jsonify({'answer': response['answer']}), 200
    except Exception as e:
        logging.error("Error in /ask endpoint: %s", traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logging.info("Starting Flask application on port 5001.")
    app.run(port=5001, debug=False)
