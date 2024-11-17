# import required libraries for similarity calculation, embeddings, and system monitoring
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
import os
import openai
import time
import psutil
from dotenv import load_dotenv

# load environment variables for secure configuration
load_dotenv()

# set openai api key from environment variables
openai.api_key = os.environ["OPENAI_API_KEY"]

# function to calculate cosine similarity between two embeddings
def compute_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# function to get embedding vector and measure performance metrics
def get_embedding_data(embed_func, text):
    start_time = time.time()
    embedding = embed_func(text)
    latency = time.time() - start_time
    memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # convert to MB
    return embedding, latency, memory_usage

# list of embedding models to evaluate
embedding_models = ["text-embedding-ada-002", "text-embedding-3-small"]

# main evaluation function for embedding models
def evaluate_embedding_model(model_name):
    print(f"Evaluating model: {model_name}")
    
    # create embedding model instance
    embedding = OpenAIEmbeddings(model=model_name)
    
    # dictionary to store performance metrics
    metrics = {}

    # get embeddings and performance metrics for each text
    embedding1, latency1, memory_usage1 = get_embedding_data(embedding.embed_query, text1)
    embedding2, latency2, memory_usage2 = get_embedding_data(embedding.embed_query, text2)
    embedding3, latency3, memory_usage3 = get_embedding_data(embedding.embed_query, text3)

    # calculate similarities between text pairs
    similarity_12 = compute_cosine_similarity(embedding1, embedding2)  # similar texts
    similarity_13 = compute_cosine_similarity(embedding1, embedding3)  # dissimilar texts
    
    # calculate average metrics
    metrics["latency"] = (latency1 + latency2 + latency3) / 3
    metrics["memory_usage_mb"] = (memory_usage1 + memory_usage2 + memory_usage3) / 3
    metrics["similarity_12"] = similarity_12
    metrics["similarity_13"] = similarity_13
    
    # print evaluation results
    print(f"Model: {model_name}")
    print(f"Average Latency: {metrics['latency']:.2f} seconds")
    print(f"Memory Usage: {metrics['memory_usage_mb']:.2f} MB")
    print(f"Similarity between text1, text2: {metrics['similarity_12']:.2f}")
    print(f"Similarity between text1, text3: {metrics['similarity_13']:.2f}")

# get input texts for comparison
text1 = input('text1:')
text2 = input('text2:')
text3 = input('text3:')

# evaluate each embedding model
for model_name in embedding_models:
    evaluate_embedding_model(model_name)
    print()