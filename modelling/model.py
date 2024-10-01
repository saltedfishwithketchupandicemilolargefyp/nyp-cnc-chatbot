from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import openai
import os
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory  # Import memory
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the variables from the .env file
CHROMA_PATH = os.getenv("CHROMA_PATH")
DATA_PATH = os.getenv("DATA_PATH")
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")

# change the retriever params
embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
retriever = db.as_retriever(search_kwargs={'k': 3})

# change the prompt template if needed, for eg if you want to reject or have a fixed
# constant reply for questions out of context
PROMPT_TEMPLATE = """You are an AI Assistant designed to provide helpful and safe information. Given the following context and previous conversations:
{context}

Conversation History:
{history}
Answer the following question and avoid giving any harmful, inappropriate, or biased content.
Respond respectfully and ethically. Do not answer inappropriate or harmful questions.
{question}

For questions that are not in the vector database, and replies that have been generated from web/training data, inform user first, then provide a reply.

Assistant:"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "history", "question"]
)

# Initialize conversation memory
memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# Create the RetrievalQA chain with memory
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=False,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT},
    memory=memory  # Add the memory module here
)

# Main interaction loop
print("Enter your question here ('Exit' to end):")
while True:
    question = input()
    if question.lower() == "exit":
        break
    response = qa.invoke({"query": question})
    result = response["result"]
    print(result)
