from langchain_community.vectorstores import (faiss, qdrant)
from qdrant_client.http import model as qdrant_client_http_model

# -------------------------------------------------------------------------------

faiss_store = faiss.FAISS

def create_faiss_index(path: str, db: faiss_store):
  db.save_local(path)

def load_faiss_index(path: str, embeddings):
  db = faiss_store.load_local(path, embeddings)
  return db

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

