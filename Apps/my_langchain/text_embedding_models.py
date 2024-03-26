from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import CohereEmbeddings

def create_text_embedding_model(name: str):
  embeddings = None
  
  if name == "openai":
    embeddings = OpenAIEmbeddings()
  elif name == "cohere":
    embeddings = CohereEmbeddings()
  
  return embeddings