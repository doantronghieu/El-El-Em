import add_packages
import config
import os
from loguru import logger

from langchain_community.vectorstores import (
  faiss, qdrant, chroma, docarray
)
from langchain_openai import OpenAIEmbeddings

from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers import (
    ContextualCompressionRetriever,
)
from langchain_core.documents import Document


import qdrant_client
from qdrant_client.http import models

from pprint import pprint
from tqdm import tqdm

# -------------------------------------------------------------------------------


def create_qdrant_index(
  docs, embeddings, use_memory=True, path=None, collection_name="my_documents"
  ):
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
  def __init__(
    self,
    qdrant_host: str,
    qdrant_api_key: str,
    configs: dict,
    collection_name: str = "",
    default_search_type: str = "similarity",  # "mmr"
    default_search_kwargs: dict = {
      "k": 6,
    },
    retriever_tool_name: str = "",
    retriever_tool_description: str = "",
  ):

    configs_qdrant = configs["vector_db"]["qdrant"]["general"]
    embeddings_name = configs["vector_db"]["embeddings"]["model"]
    embeddings_params = configs["vector_db"]["embeddings"][embeddings_name]
    embeddings_model = embeddings_params["model"]
    embeddings_size = configs["vector_db"]["embeddings"]["size"][embeddings_model]

    self.client = qdrant_client.QdrantClient(
      location=qdrant_host,
      api_key=qdrant_api_key,
      prefer_grpc=configs_qdrant["prefer_grpc"],
    )

    self.collection_name = collection_name
    client_collections = str(self.client.get_collections())

    vectors_config = models.VectorParams(
      size=embeddings_size,
      distance=configs_qdrant["distance"],
    )

    # *---------------------------------------------------------------------
    if self.collection_name in client_collections:
      logger.info(f"Found collection: `{self.collection_name}`.")
    else:
      self.client.create_collection(
        collection_name=self.collection_name,
        vectors_config=vectors_config,
      )
      logger.info(f"Collection: `{self.collection_name}` created.")

    # *---------------------------------------------------------------------

    logger.info(
      f"`{self.collection_name}` - Embeddings: {embeddings_name} - {embeddings_params}, {embeddings_size}")
    if embeddings_name == "openai":
      self.embeddings = OpenAIEmbeddings(**embeddings_params)
    elif embeddings_name == "cohere":
      self.embeddings = None
      logger.warning("Not implemented yet")

    # *---------------------------------------------------------------------

    self.vector_store = qdrant.Qdrant(
      client=self.client,
      collection_name=self.collection_name,
      embeddings=self.embeddings,
      content_payload_key=configs_qdrant["content_payload_key"],
      metadata_payload_key=configs_qdrant["metadata_payload_key"],
    )

    # *---------------------------------------------------------------------
    if embeddings_name == "cohere":
      logger.info("`{self.collection_name}` - Retriever: Cohere")
      logger.warning("Not implemented yet")
      self.retriever = None
    else:
      logger.info(f"`{self.collection_name}` - Retriever: Vectorstore")
      self.retriever = self.vector_store.as_retriever(
        search_type=default_search_type,
        search_kwargs=default_search_kwargs,
      )

    # *---------------------------------------------------------------------
    self.retriever_tool = create_retriever_tool(
      retriever=self.retriever,
      name=retriever_tool_name,
      description=retriever_tool_description,
    )

  def add_documents(self, docs: list[Document]):
    for doc in tqdm(docs):
      self.vector_store.add_documents([doc])

  def invoke_retriever(self, query, **kwargs):
    return self.retriever.invoke(query, **kwargs)
