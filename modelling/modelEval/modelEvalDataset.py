# same as modelWithConvoHist.py, just that tracing is enabled here for offline evaluation.
# import required libraries for langchain, openai, and other utilities
from langsmith import Client
from langchain.callbacks import LangChainTracer
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

# load environment variables from .env file for secure configuration
load_dotenv()

# get environment variables for paths and api keys
CHROMA_PATH = os.getenv("CHROMA_PATH")
DATA_PATH = os.getenv("DATA_PATH")
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

# setting up tracing project
callbacks = [
LangChainTracer(
  project_name= os.getenv("LANGCHAIN_PROJ_NAME"),
  client=Client(
    api_url="https://api.smith.langchain.com",
    api_key=os.getenv("LANGCHAIN_API_KEY")
  )
)
]

# initialize the language model with temperature for creativity
llm = ChatOpenAI(temperature=0.8, model="gpt-4o-mini")

# set up the vector database with embeddings for document retrieval
embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

# define template for generating multiple queries from a single user question
# this helps in getting more relevant search results
multi_query_template = PromptTemplate(
    template=(
        "The user has asked a question: {question}.\n"
        "1. First, check if it is a complex question or multiple related questions. "
        "If yes, split the query into distinct questions if there are multiple and move on to point 2."
        "Else, just return the question, and break.\n"
        "2. Then, for each distinct question, generate 3 rephrasings that would return "
        "similar but slightly different relevant results.\n"
        "Return each question on a new line with its rephrasings.\n"
    ),
    input_variables=["question"],
)

# create a retriever that generates multiple queries for better search coverage
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    llm=llm,
    prompt=multi_query_template
)

# define system prompt for contextualizing questions based on chat history
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# create prompt template that includes chat history for context
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# create a retriever that's aware of conversation history
history_aware_retriever = create_history_aware_retriever(
    llm, multiquery_retriever, contextualize_q_prompt
)

# define the system prompt for the main question answering task
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use ONLY the following pieces of retrieved context to answer the questions. "
    "Answer the following question and avoid giving any harmful, inappropriate, or biased content. "
    "Respond respectfully and ethically. Do not answer inappropriate or harmful questions. "
    "If the answer does not exist in the vector database, "
    "nicely inform the user that you cannot answer questions that are not in the NYP CNC database. "
    "Keep the answer concise."
    "\n\n"
    "{context}"
)

# create the main qa prompt template including chat history
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# create the chain that combines document retrieval and question answering
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# define the state type for managing conversation state
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

# function to process user input and generate response while maintaining state
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

# create a workflow graph for managing the conversation flow
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# set up memory to persist conversation state
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# configuration for the conversation thread
config = {"configurable": {"thread_id": "abc123"}}

# main loop for user interaction
print("Enter your question here ('Exit' to end):")
while True:
    question = input()
    if question.lower() == "exit":
        break
    # process the user's question and get response
    response = app.invoke({"input": question}, config=config)
    print(response["answer"])
    
    print()  
    print("Enter your question here ('Exit' to end):")