# required imports for building a retrieval-augmented generation (rag) system
# core dependencies for time tracking and environment variables
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

# load environment configuration from .env file for secure credential management
load_dotenv()

# retrieve environment variables for configuration
CHROMA_PATH = os.getenv("CHROMA_PATH")          # path to vector database storage
DATA_PATH = os.getenv("DATA_PATH")              # path to source documents
openai.api_key = os.getenv("OPENAI_API_KEY")    # openai api authentication
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")   # model for generating embeddings

# load model configurations from yaml file
# allows for easy modification of model parameters without changing code
with open("config.yaml", "r") as config_file:
    config_data = yaml.safe_load(config_file)

# initialize the vector database and retriever
# setup embedding model for converting text to vectors
embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
# initialize chromadb with the specified embedding function
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
# create retriever that fetches top 3 most relevant documents
retriever = db.as_retriever(search_kwargs={'k': 3})

# define template for generating multiple versions of the input query
# this helps improve retrieval by considering different phrasings
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

# system prompt for processing chat history and current question
# helps maintain context across conversation turns
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# template for handling conversation history and new questions
# combines system instructions with chat history and user input
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# main system prompt for the qa system
# defines behavior, constraints, and response format
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

# qa prompt template combining system prompt, chat history, and user input
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# process each model specified in the configuration
for model in config_data['models']:
    # initialize language model with specified parameters
    llm = ChatOpenAI(temperature=model['temperature'], model=model['name'], streaming=True)

    # create retriever that generates multiple query variations
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        prompt=multi_query_template
    )

    # create retriever that incorporates conversation history
    history_aware_retriever = create_history_aware_retriever(
        llm, multiquery_retriever, contextualize_q_prompt
    )

    # create chain for processing retrieved documents and generating answers
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # combine retrieval and question answering into a single chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # define the structure for maintaining conversation state
    class State(TypedDict):
        input: str                                                    # user's question
        chat_history: Annotated[Sequence[BaseMessage], add_messages]  # conversation history
        context: str                                                  # retrieved context
        answer: str                                                   # generated response

    # function to process input and generate response using the rag chain
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

    # setup the conversation workflow
    workflow = StateGraph(state_schema=State)
    workflow.add_edge(START, "model")           # connect starting point to model
    workflow.add_node("model", call_model)      # add model processing node

    # initialize system for saving conversation state
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)

    # configuration for the conversation thread
    config = {"configurable": {"thread_id": "abc123"}}

    # main loop for handling user interactions
    print('-'*150)
    print(f'Using {model["name"]}')
    print('='*150)

    # continuous interaction loop
    while True:
        print("Enter your question here ('Exit' to end):")
        question = input()
        if question.lower() == "exit":
            break

        # track start time for performance monitoring
        start_time = time.time()

        # process the question through the rag system
        response = app.invoke({"input": question}, config=config)

        # calculate processing time
        end_time = time.time()
        response_time = end_time - start_time

        # output results and performance metrics
        print(f"Response: {response['answer']}")
        print(f"Response Time: {response_time:.2f} seconds")
        print(response)

        print('='*150)