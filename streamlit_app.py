# streamlit_app.py bryant part

import streamlit as st
import requests

st.title("AI Chatbot Helper for NYP CNC")

# File Upload Section
st.header("Upload a File")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "xlsx", "pptx"])
if uploaded_file is not None:
    files = {"file": uploaded_file}
    response = requests.post("http://127.0.0.1:5001/upload", files=files)
    if response.status_code == 200:
        st.success("File uploaded and processed successfully!")
    else:
        st.error(f"File upload failed: {response.json().get('error')}")

# Question Answering Section
st.header("Ask a Question")
question = st.text_input("Enter your question")
if st.button("Submit Question") and question:
    payload = {"question": question}
    response = requests.post("http://127.0.0.1:5001/ask", json=payload)
    if response.status_code == 200:
        st.write("Answer:")
        st.write(response.json().get("answer"))
    else:
        st.error("Failed to get an answer: " + response.json().get("error"))
