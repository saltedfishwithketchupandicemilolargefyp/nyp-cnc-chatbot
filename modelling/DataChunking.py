import os
import numpy as np
import nltk
nltk.download('punkt')

# loading the extracted text
from langchain.document_loaders import TextLoader

# splitting text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# for embedding the chunks
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# to store all embedded chunks into a vector store
from langchain_community.vectorstores import FAISS

# hugging face llm pipeline
from langchain_community.llms import HuggingFacePipeline

# for question answering
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
#---- Set OpenAI API key 
# Change environment variable name from "OPENAI_API_KEY" to the name given in 
# your .env file.


def main():
    generate_data_vectorstore()


def generate_data_vectorstore():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_faiss(chunks)


def load_documents():
    # loading the extracted_text.txt file
    loader = TextLoader('./extracted_text.txt')
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True)
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    
    # for testing and reviewing the different chunks
    # document = chunks[10]
    # print(document.page_content)
    # print(document.metadata)

    return chunks

# embeddings model

huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",  # alternatively use "sentence-transformers/all-MiniLM-l6-v2" for a light and faster experience.
    model_kwargs={'device':'cpu'}, 
    encode_kwargs={'normalize_embeddings': True}
)


def save_to_faiss(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, hf, persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()



