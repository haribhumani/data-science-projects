import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def load_dataset(filename):
    # Load the SQuAD dataset and extract paragraphs
    df = pd.read_csv(filename)
    paragraphs = df['Plot'].to_list()
    titles = df['Title'].to_list()
    return paragraphs, titles

def generate_dense_vectors(transformer, tokenizer, paragraphs):
    vectors = transformer.encode(paragraphs)
    save_dense_vectors(vectors)
    return vectors

def save_dense_vectors(vectors, file_name="dense_vectors.pt"):
    torch.save(vectors, file_name)

def load_dense_vectors(file_name):
    return torch.load(file_name)

#def retrieve(transformer, tokenizer, query, dense_vectors, paragraphs, top_k=5):
def retrieve(transformer, tokenizer, query, dense_vectors, top_k=5):
    
    query_vector = generate_dense_vectors(transformer, tokenizer, [query])
    
    similarity_scores = cosine_similarity(query_vector, dense_vectors)[0]

    top_indices = similarity_scores.argsort()[-top_k:][::-1]

    indices_scores = [(ind, similarity_scores[ind])  for ind in top_indices]
    
    return indices_scores