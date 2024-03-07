from loguru import logger

from langchain_community.vectorstores import (faiss, qdrant, chroma)

from langchain_openai import OpenAIEmbeddings

import qdrant_client
from qdrant_client.http import models
from qdrant_client.http import model as qdrant_client_http_model

from pprint import pprint
# -------------------------------------------------------------------------------

faiss_store = faiss.FAISS

def create_faiss_index(path: str, db: faiss_store):
  db.save_local(path)

def load_faiss_index(path: str, embeddings):
  db = faiss_store.load_local(path, embeddings)
  return db

# -------------------------------------------------------------------------------

chroma_store = chroma.Chroma

#-------------------------------------------------------------------------------

qdrant_store = qdrant.Qdrant

def create_qdrant_index(docs, embeddings, use_memory=True, path=None, 
                        collection_name="my_documents"):
  """
  Create a Qdrant index and return the Qdrant object.

  Parameters:
  - docs: List of documents to index
  - embeddings: List of corresponding document embeddings
  - use_memory: Flag to indicate whether to use in-memory storage (default is True)
  - path: Path for on-disk storage (if using file-based storage)
  - collection_name: Name for the Qdrant collection

  Returns:
  - Qdrant object
  """
  if use_memory:
    qdrant_instance = qdrant.Qdrant.from_documents(
      docs,
      embeddings,
      location=":memory:",
      collection_name=collection_name,
    )
  else:
    qdrant_instance = qdrant.Qdrant.from_documents(
      docs,
      embeddings,
      path=path,
      collection_name=collection_name,
    )

  return qdrant_instance


class QdrantWrapper:
  def __init__(self,
                collection_name,
                qdrant_host,
                qdrant_api_key,
                prefer_grpc=True,
                size=1536,
                distance=models.Distance.COSINE,
                embeddings=OpenAIEmbeddings(),
                chain=None,
                default_search_type="similarity",  # "mmr"
                default_search_kwargs={
                    "k": 6,
                },
                content_payload_key: str = 'page_content',
                metadata_payload_key: str = 'metadata',
                ):
      self.client = qdrant_client.QdrantClient(
        location=qdrant_host,
        api_key=qdrant_api_key,
        prefer_grpc=prefer_grpc,
      )

      self.content_payload_key = content_payload_key
      self.metadata_payload_key = metadata_payload_key

      self.collection_name = collection_name
      client_collections = str(self.client.get_collections())
      
      vectors_config = models.VectorParams(
        size=size, 
        distance=distance,
      )
      
      if self.collection_name in client_collections:
        logger.info(f"Found collection: `{self.collection_name}`.")
      else:
        self.client.create_collection(
          collection_name=self.collection_name,
          vectors_config=vectors_config,
        )
        logger.info(f"Collection: `{self.collection_name}` created.")

      self.embeddings = embeddings
      self.vector_store = qdrant.Qdrant(
        client=self.client,
        collection_name=self.collection_name,
        embeddings=self.embeddings,
        content_payload_key=content_payload_key,
        metadata_payload_key=metadata_payload_key,
      )

      self.default_search_kwargs = default_search_kwargs
      self.default_search_type = default_search_type

      self.retriever = self.vector_store.as_retriever(
        search_type=default_search_type,
        search_kwargs=default_search_kwargs,
      )

      self.chain = chain

  def invoke_query(self, query):
    response = self.chain.invoke(query)
    pprint(response['result'])

  def invoke_retriever(self, query, **kwargs):
    return self.retriever.invoke(query, **kwargs)
