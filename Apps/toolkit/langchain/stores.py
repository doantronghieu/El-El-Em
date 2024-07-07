import add_packages
import os
from my_configs import constants

import logging
from loguru import logger
from tqdm import tqdm
import typing_inspect
from typing import List, Dict, Any, Optional, Literal, Union


from langchain_community.vectorstores import chroma, docarray, faiss, qdrant
from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter
from langchain_community.retrievers import BM25Retriever
from langchain_community.utilities import SQLDatabase

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings

from langchain.retrievers import (
  ContextualCompressionRetriever,
  EnsembleRetriever,
  MultiQueryRetriever,
  RePhraseQueryRetriever,
  SelfQueryRetriever
)
from langchain.retrievers.document_compressors import (
  DocumentCompressorPipeline,
  EmbeddingsFilter,
  LLMChainExtractor,
  LLMChainFilter
)
from langchain.tools.retriever import create_retriever_tool

from langchain_cohere import CohereEmbeddings, CohereRerank

# Qdrant
from langchain_qdrant import Qdrant
import qdrant_client
from qdrant_client.http import models

# MongoDB
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

# Logging setup
if os.getenv("IN_PROD"):
  logging.basicConfig()
  logging.getLogger("langchain.retrievers.re_phraser").setLevel(logging.INFO)
  logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

#-------------------------------------------------------------------------------

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
    qdrant_instance = Qdrant.from_documents(
      docs,
      embeddings,
      location=":memory:",
      collection_name=collection_name,
    )
  else:
    qdrant_instance = Qdrant.from_documents(
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

    # *---------------------------------------------------------------------
    if self.collection_name in client_collections:
      logger.info(f"Found collection: `{self.collection_name}`.")
    else:
      self.client.create_collection(
        collection_name=self.collection_name,
        vectors_config=models.VectorParams(
          size=embeddings_size,
          distance=configs_qdrant["distance"],
        ),
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

    self.vector_store = Qdrant(
      client=self.client,
      collection_name=self.collection_name,
      embeddings=self.embeddings,
      content_payload_key=configs_qdrant["content_payload_key"],
      metadata_payload_key=configs_qdrant["metadata_payload_key"],
    )

    # *---------------------------------------------------------------------
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

#-------------------------------------------------------------------------------

class MongoStore:
		def __init__(
				self,
				mongodb_atlas_cluster_uri: str,
				db_name: str,
				collection_name: str,
				index_name: str,
				dimensions: int,
				embeddings: Optional[Any] = None,
				configs: Dict = None,
				default_search_type: str = "similarity",
				default_search_kwargs: Dict = {"k": 6},
				retriever_tool_name: str = "",
				retriever_tool_description: str = "",
		):
				self.client = MongoClient(mongodb_atlas_cluster_uri)
				self.db_name = db_name
				self.collection_name = collection_name
				self.index_name = index_name
				self.dimensions = dimensions
				self.collection = self.client[db_name][collection_name]
				
				# Set up embeddings
				self.embeddings = embeddings if embeddings else OpenAIEmbeddings(disallowed_special=())
				
				# Set up vector store
				self.vector_store = MongoDBAtlasVectorSearch(
						collection=self.collection,
						index_name=self.index_name,
						embedding=self.embeddings,
						relevance_score_fn="cosine",
				)
				
				# Set up retriever
				self.retriever = self.vector_store.as_retriever(
						search_type=default_search_type,
						search_kwargs=default_search_kwargs,
				)
				
				# Set up retriever tool
				if retriever_tool_name and retriever_tool_description:
						self.retriever_tool = create_retriever_tool(
								retriever=self.retriever,
								name=retriever_tool_name,
								description=retriever_tool_description,
						)
				else:
						self.retriever_tool = None

		def create_index(self):
				"""
				Not available in free-tier.
				https://www.mongodb.com/docs/atlas/atlas-vector-search/ai-integrations/langchain/#create-the-atlas-vector-search-index
				"""
				try:
						search_index_model = SearchIndexModel(
								definition={
										"fields": [
												{
														"type": "vector",
														"numDimensions": self.dimensions,
														"path": "embedding",
														"similarity": "cosine"
												},
												{
														"type": "filter",
														"path": "page"
												},
										]
								},
								name=self.index_name,
								type="vectorSearch",
						)

						result = self.collection.create_search_index(model=search_index_model)
						logger.info(f"Index '{self.index_name}' created successfully. Result: {result}")
						return result
				except Exception as e:
						if "already exists" in str(e):
								logger.info(f"Index '{self.index_name}' already exists.")
						else:
								logger.error(f"Error creating index: {e}")
						return None

		def add_documents(self, docs: List[Document]):
				for doc in tqdm(docs):
						self.vector_store.add_documents([doc])

		def invoke_retriever(self, query, **kwargs):
				result: List[Document] = self.retriever.invoke(query, **kwargs)
				result = [Document(res.page_content) for res in result]
				return result

"""
MONGODB_DB_NAME = "db_langchain"
MONGODB_COLLECTION_NAME = "coll_langchain"
MONGODB_ATLAS_VECTOR_SEARCH_INDEX_NAME = "idx_langchain"

mongo_store = MongoStore(
	mongodb_atlas_cluster_uri=os.getenv("MONGODB_ATLAS_CLUSTER_URI"),
	db_name=MONGODB_DB_NAME,
	collection_name=MONGODB_COLLECTION_NAME,
	index_name=MONGODB_ATLAS_VECTOR_SEARCH_INDEX_NAME,
	dimensions=1536,  # Set this to match your embedding size
	retriever_tool_name="retriever_mongodb",
	retriever_tool_description="Useful for retrieving information from MongoDB Atlas vector store."
)

"""
#-------------------------------------------------------------------------------

TypeRetriever = Literal[
  'base', 'SelfQueryRetriever', 'MultiQueryRetriever', 'CohereRerank', 
  'BM25Retriever', 'RePhraseQueryRetriever',
]
lst_type_retriever = list(typing_inspect.get_args(TypeRetriever))

def create_portion(input_list):
  length = len(input_list)
  output_value = 1 / length
  output_list = [output_value] * length
  return output_list

def create_retriever(
  llm: Union[BaseChatModel, None] = None,
  vectorstore: Union[VectorStore, None] = None,
  embeddings: Union[Embeddings, None] = None,
  retriever_types: list[TypeRetriever] = [
    'base', 'MultiQueryRetriever', 'RePhraseQueryRetriever',
  ],
  compressor_types: list[Literal[
    'EmbeddingsRedundantFilter', 'EmbeddingsFilter', 'LLMChainFilter', 
    'LLMChainExtractor',
  ]] = [],
  search_type: Literal['mmr', 'similarity'] = "mmr",
  search_kwargs: dict = {
    "k": 6,
  },
  document_content_description: Union[str, None] = None,
  metadata_field_info: Union[list, None] = None,
):
  my_retrievers = []
  my_compressors = []
  
  #*----------------------------------------------------------------------------
  retriever_base = vectorstore.as_retriever(
    search_type=search_type,
    search_kwargs=search_kwargs,
  )
  
  if "base" in retriever_types:
    my_retrievers.append(retriever_base)
  if "SelfQueryRetriever" in retriever_types:
    my_retrievers.append(SelfQueryRetriever.from_llm(
      llm=llm,
      vectorstore=vectorstore,
      document_contents=document_content_description,
      metadata_field_info=metadata_field_info,
      verbose=True,
    ))
  if "MultiQueryRetriever" in retriever_types:
    my_retrievers.append(MultiQueryRetriever.from_llm(
      retriever=retriever_base, llm=llm,
    ))
  if "CohereRerank" in retriever_types:
    logger.warning(f"Remember to use CohereEmbeddings for Vectorstore.")
    embeddings = CohereEmbeddings(
      model=constants.EMBEDDINGS["COHERE"]["EMBED-MULTILINGUAL-V3.0"]
    )
    my_retrievers.append(ContextualCompressionRetriever(
      base_compressor=CohereRerank(), base_retriever=retriever_base,
    ))
  if "RePhraseQueryRetriever" in retriever_types:
    my_retrievers.append(RePhraseQueryRetriever.from_llm(
      retriever=retriever_base, llm=llm
    ))
  if "BM25Retriever" in retriever_types:
    my_retrievers.append(BM25Retriever()) # todo
    
  #*----------------------------------------------------------------------------

  if "ContextualCompressionRetriever" in retriever_types:
    if "EmbeddingsRedundantFilter" in compressor_types:
      filter_embeddings_redundant = EmbeddingsRedundantFilter(embeddings=embeddings)
      my_compressors.append(filter_embeddings_redundant)
    if "EmbeddingsFilter" in compressor_types:
      filter_embeddings_relevant = EmbeddingsFilter(
        embeddings=embeddings, similarity_threshold=0.75,
      )
      my_compressors.append(filter_embeddings_relevant)
    
    if "LLMChainFilter" in compressor_types:
      filter_llmchain = LLMChainFilter.from_llm(llm)
      my_compressors.append(filter_llmchain)
    
    if "LLMChainExtractor" in compressor_types:
      extractor_llmchain = LLMChainExtractor.from_llm(llm)
      my_compressors.append(extractor_llmchain)
      
    compressor_pipeline = DocumentCompressorPipeline(
      transformers=my_compressors,
    )

    retriever_contextual_compression = ContextualCompressionRetriever(
      base_compressor=compressor_pipeline, base_retriever=retriever_base,
    )
    
    my_retrievers.append(retriever_contextual_compression)
  
  logger.info(f"Retrievers: {retriever_types}")
  retriever_ensemble = EnsembleRetriever(
    retrievers=my_retrievers,
    weights=create_portion(my_retrievers),
  )

  return retriever_ensemble
