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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

def main():
    documents = load_text(DATA_PATH)
    chunks = split_text(documents)
    split_chunked = split_list(chunks, 166)
    create_db(split_chunked)

# Loading the extracted_text.txt file
def load_text(DATA_PATH):
    loader = TextLoader(DATA_PATH, encoding='utf-8')
    documents = loader.load()
    return documents

# Splitting text into chunks
def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=400,
        length_function=len,
        add_start_index=True
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # printing some chunks for visualisation, can be removed if not needed
    print('Example Chunks:')
    print(chunks[1])
    print('='*100)
    print(chunks[2])
    print('='*100)
    print(chunks[3])

    return chunks

def split_list(chunks, batch_size):
    for i in range(0, len(chunks), batch_size):
        yield chunks[i:i + batch_size]


def create_db(split_chunks):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    
    for chunk in split_chunks:
        db = Chroma.from_documents(
            documents=chunk,
            embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
            persist_directory=CHROMA_PATH,
        )

        print(f"Added {len(chunk)} chunks to the database.")


if __name__ == "__main__":
    main()
