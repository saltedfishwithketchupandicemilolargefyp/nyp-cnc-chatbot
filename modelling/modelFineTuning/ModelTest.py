# Required imports
import time
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
import openai
import os
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
from typing import Sequence
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import PromptTemplate
import yaml

# Load environment variables from .env file
load_dotenv()
# Access the variables from the .env file
CHROMA_PATH = os.getenv("CHROMA_PATH")
DATA_PATH = os.getenv("DATA_PATH")
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# Load models configuration from config.yaml
with open("config.yaml", "r") as config_file:
    config_data = yaml.safe_load(config_file)

# Setup retriever
embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
retriever = db.as_retriever(search_kwargs={'k': 3})

# Define the multi-query generation template for retriever
multi_query_template = PromptTemplate(
    template=(
        "The user has asked a complex question or multiple related questions: {question}.\n"
        "1. First, split the query into distinct questions if there are multiple.\n"
        "2. Then, for each distinct question, generate 3 rephrasings that would return "
        "similar but slightly different relevant results.\n"
        "Return each question on a new line with its rephrasings.\n"
    ),
    input_variables=["question"],
)

# Prompt to get the chat history context for follow-up questions
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# History context prompt template
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# System prompt for question answering
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the questions. "
    "Answer the following question and avoid giving any harmful, inappropriate, or biased content. "
    "Respond respectfully and ethically. Do not answer inappropriate or harmful questions. "
    "If the answer does not exist in the vector database, "
    "nicely inform the user that you cannot answer questions that are not in the NYP CNC database. "
    "Keep the answer concise."
    "\n\n"
    "{context}"
)

# QA prompt template
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Loop through each model in config.yaml
for model in config_data['models']:
    # Create the LLM for this iteration
    llm = ChatOpenAI(temperature=model['temperature'], model=model['name'], streaming=True)

    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        prompt=multi_query_template
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, multiquery_retriever, contextualize_q_prompt
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    class State(TypedDict):
        input: str
        chat_history: Annotated[Sequence[BaseMessage], add_messages]
        context: str
        answer: str

    # Node to run the RAG chain and update the state
    def call_model(state: State):
        response = rag_chain.invoke(state)
        return {
            "chat_history": [
                HumanMessage(state["input"]),
                AIMessage(response["answer"]),
            ],
            "context": response["context"],
            "answer": response["answer"],
        }

    # Graph definition
    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_model)

    # MemorySaver to checkpoint state
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    config = {"configurable": {"thread_id": "abc123"}}

    # Main interaction loop
    print('-'*150)
    print(f'Using {model["name"]}')
    print('='*150)

    while True:
        print("Enter your question here ('Exit' to end):")
        question = input()
        if question.lower() == "exit":
            break

        # Measure response time
        start_time = time.time()

        # Invoke the model
        response = app.invoke({"input": question}, config=config)

        # Measure end time
        end_time = time.time()

        # Calculate response time
        response_time = end_time - start_time

        # Display response and performance metrics
        print(f"Response: {response['answer']}")
        print(f"Response Time: {response_time:.2f} seconds")
        print(response)

        print('='*150)