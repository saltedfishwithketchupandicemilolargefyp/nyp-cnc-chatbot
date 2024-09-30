from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import openai
import os
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the variables from the .env file
CHROMA_PATH = os.getenv("CHROMA_PATH")
DATA_PATH = os.getenv("DATA_PATH")
openai.api_key = os.getenv("OPENAI_API_KEY")
embedding_model = os.getenv("EMBEDDING_MODEL")


# def get_embedding_model(model_name):
#     if model_name == "OpenAIEmbeddings":
#         return OpenAIEmbeddings()  # Ensure this returns an instance
#     # You can add other embedding models here
#     else:
#         raise ValueError(f"Unknown embedding model: {model_name}")


def main():
    documents = load_text(DATA_PATH)
    chunks = split_text(documents)
    create_db(chunks)

# Loading the extracted_text.txt file
def load_text(DATA_PATH):
    loader = TextLoader(DATA_PATH, encoding='utf-8')
    documents = loader.load()
    return documents

# Splitting text into chunks
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=400,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks

def create_db(chunks):
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


if __name__ == "__main__":
    main()
