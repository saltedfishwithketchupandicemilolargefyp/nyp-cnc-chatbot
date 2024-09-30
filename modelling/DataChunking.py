
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 

from dotenv import load_dotenv
import os
import shutil

# Load environment variables from .env file
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "modelling\data"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    return documents

# Splitting text into chunks
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks

# embeddings model
model_name = "sentence-transformers/all-mpnet-base-v2"

hf = HuggingFaceEmbeddings(model_name=model_name)


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents with the specified embedding function
    # embeddings = get_embedding_model(embedding_model)  # Retrieve the embedding model
    db = Chroma.from_documents(chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH)

    # Batch size limit
    max_batch_size = 166

    # Add chunks in smaller batches
    for i in range(0, len(chunks), max_batch_size):
        batch = chunks[i:i + max_batch_size]
        db.add_texts(batch)  # Adjust this line if necessary.

    print(f"Added {len(chunks)} chunks to the database.")
    return db


if __name__ == "__main__":
    main()
