# import necessary libraries for rag implementation and evaluation
from langsmith import Client
from langsmith.evaluation import evaluate
from langchain import hub
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
from pprint import pprint

# setup environment and configuration
load_dotenv()
CHROMA_PATH = os.getenv("CHROMA_PATH")
DATA_PATH = os.getenv("DATA_PATH")
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))

# initialize the language model with temperature 0.8 for some creativity
llm = ChatOpenAI(temperature=0.8, model="gpt-4o-mini")

# set up vector database with embeddings
embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)

# define template for generating multiple query variations
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

# create retriever that generates multiple queries for better search results
multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    llm=llm,
    prompt=multi_query_template
)

# prompt to handle chat history context
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

# set up chat history aware retrieval system
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# create retriever that considers conversation history
history_aware_retriever = create_history_aware_retriever(
    llm, multiquery_retriever, contextualize_q_prompt
)

# define system prompt for the qa system
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

# set up the complete rag pipeline
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# define state management for conversation
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str

# function to process input and generate response
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

# set up workflow management
workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# evaluation setup
config = {"configurable": {"thread_id": "abc123"}}

# prompts for evaluation
grade_prompt_hallucinations = hub.pull("langchain-ai/rag-answer-hallucination")
grade_prompt_answer_helpfulness = hub.pull("langchain-ai/rag-answer-helpfulness")

# llm for grading
grader_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# loading in the dataset
dataset_name = "RAG Chatbot Dataset"
examples = list(client.list_examples(dataset_name=dataset_name))

# prediction functions for evaluation
def predict_rag_answer(example):
    """generate answers for basic evaluation"""
    question = example["input"]
    response = app.invoke({"input": question}, config=config)
    return {"answer": response["answer"]}

def predict_rag_answer_with_context(example: dict):
    """generate answers with context for hallucination checking"""
    question = example["input"]
    response = app.invoke({"input": question}, config=config)
    return {
        "answer": response["answer"], 
        "context": response["context"]
    }


# evaluation functions
def answer_helpfulness_evaluator(run, example):
    """evaluate how helpful the answers are"""
    input_question = example.inputs["input"]
    prediction = example.outputs["answer"]
    answer_grader = grade_prompt_answer_helpfulness | grader_llm
    result = answer_grader.invoke({
        "question": input_question,
        "student_answer": prediction
    })
    return {"key": "answer_helpfulness_score", "score": result["Score"]}

def answer_hallucination_evaluator(run, example):
    """check if answers contain information not present in context"""
    page_contents = [doc['page_content'] for doc in example.outputs["context"]]
    contexts = page_contents
    prediction = example.outputs["answer"]
    answer_grader = grade_prompt_hallucinations | grader_llm
    result = answer_grader.invoke({
        "documents": contexts,
        "student_answer": prediction
    })
    return {"key": "answer_hallucination", "score": result["Score"]}

# run evaluations
helpfulness_results = evaluate(
    predict_rag_answer,
    data=examples,
    evaluators=[answer_helpfulness_evaluator],
    experiment_prefix="rag-answer-helpfulness",
    metadata={"version": "helpfulness-evaluation-v1"}
)

hallucination_results = evaluate(
    predict_rag_answer_with_context, 
    data=examples,
    evaluators=[answer_hallucination_evaluator],
    experiment_prefix="rag-answer-hallucination",
    metadata={"version": "hallucination-evaluation-v1"}
)