from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import OpenAIEmbeddings
import os
import openai
import time
import psutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]


def compute_cosine_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Function to get embedding, latency, and memory usage
def get_embedding_data(embed_func, text):
    start_time = time.time()
    embedding = embed_func(text)
    latency = time.time() - start_time
    memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # in MB
    return embedding, latency, memory_usage


# Initialize embeddings for different models
embedding_models = ["text-embedding-ada-002", "text-embedding-3-small"]

# Evaluation function
def evaluate_embedding_model(model_name):
    print(f"Evaluating model: {model_name}")
    
    # Initialize the embedding model
    embedding = OpenAIEmbeddings(model=model_name)
    
    # Track performance metrics
    metrics = {}

    # Evaluate text1 and text2 (similar), and text1 and text3 (dissimilar)
    embedding1, latency1, memory_usage1 = get_embedding_data(embedding.embed_query, text1) # Call get_embedding_data instead of evaluate_embedding_model
    embedding2, latency2, memory_usage2 = get_embedding_data(embedding.embed_query, text2) # Call get_embedding_data instead of evaluate_embedding_model
    embedding3, latency3, memory_usage3 = get_embedding_data(embedding.embed_query, text3) # Call get_embedding_data instead of evaluate_embedding_model

    # Calculate cosine similarities
    similarity_12 = compute_cosine_similarity(embedding1, embedding2)  # Should be high
    similarity_13 = compute_cosine_similarity(embedding1, embedding3)  # Should be low
    
    metrics["latency"] = (latency1 + latency2 + latency3) / 3
    metrics["memory_usage_mb"] = (memory_usage1 + memory_usage2 + memory_usage3) / 3
    metrics["similarity_12"] = similarity_12
    metrics["similarity_13"] = similarity_13
    

    # Print metrics
    print(f"Model: {model_name}")
    print(f"Average Latency: {metrics['latency']:.2f} seconds")
    print(f"Memory Usage: {metrics['memory_usage_mb']:.2f} MB")
    print(f"Similarity between text1, text2: {metrics['similarity_12']:.2f}")
    print(f"Similarity between text1, text3: {metrics['similarity_13']:.2f}")


text1 = input('text1:')
text2 = input('text2:')
text3 = input('text3:')

for model_name in embedding_models:
    evaluate_embedding_model(model_name)
    print()