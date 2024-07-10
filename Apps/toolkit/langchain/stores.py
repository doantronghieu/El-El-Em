import os, dotenv
import add_packages
from my_configs import constants
import logging
from loguru import logger
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Literal, Union, TypeAlias

from langchain_community.vectorstores import chroma, docarray, faiss, qdrant
from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter
from langchain_community.retrievers import BM25Retriever
from langchain_community.utilities import SQLDatabase

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings

from langchain.retrievers import (
  ContextualCompressionRetriever, EnsembleRetriever, MultiQueryRetriever,
  RePhraseQueryRetriever, SelfQueryRetriever
)
from langchain.retrievers.document_compressors import (
  DocumentCompressorPipeline, EmbeddingsFilter, LLMChainExtractor, LLMChainFilter
)
from langchain.tools.retriever import create_retriever_tool

from langchain_cohere import CohereEmbeddings, CohereRerank
from langchain_qdrant import Qdrant
import qdrant_client
from qdrant_client.http import models
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

# Logging setup
if not os.getenv("IN_PROD"):
  logging.basicConfig()
  logging.getLogger("langchain.retrievers.re_phraser").setLevel(logging.INFO)
  logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

dotenv.load_dotenv()
#===============================================================================

TypeRetriever: TypeAlias = Literal[
  'base', 'SelfQueryRetriever', 'MultiQueryRetriever', 'CohereRerank',
  'BM25Retriever', 'RePhraseQueryRetriever',
]

TypeCompressor: TypeAlias = Literal[
  'EmbeddingsRedundantFilter', 'EmbeddingsFilter', 'LLMChainFilter',
  'LLMChainExtractor',
]

TypeSearch: TypeAlias = Literal['mmr', 'similarity']

EmbeddingsProvider: TypeAlias = Literal["openai", "cohere"]
EmbeddingsModel: TypeAlias = Literal["text-embedding-3-large", "text-embedding-ada-002"]
QdrantTypeDistance: TypeAlias = Literal["Dot", "Cosine", "Euclid", "Manhattan"]

#===============================================================================

def get_embeddings_size(model: EmbeddingsModel) -> int:
  if model == "text-embedding-3-large": return 3072
  if model == "text-embedding-ada-002": return 1536

def create_embeddings(
  provider: EmbeddingsProvider = "openai",
  model: EmbeddingsModel = "text-embedding-ada-002"
) -> Embeddings:
  try:
    if provider == "openai":
      embeddings = OpenAIEmbeddings(model=model)
    elif provider == "cohere":
      embeddings = CohereEmbeddings(model=model)
    else:
      raise ValueError(f"Unsupported embeddings provider: {provider}")
    logger.info(f"[Embeddings] {provider}. Model: {model}")
    return embeddings
  except Exception as e:
    logger.error(f"[Embeddings] Error: {e}")
    raise

def create_portion(input_list):
  length = len(input_list)
  output_value = 1 / length
  output_list = [output_value] * length
  return output_list

def create_retriever(
  llm: Union[BaseChatModel, None],
  vectorstore: Union[VectorStore, None],
  embeddings: Union[Embeddings, None],
  retriever_types: list[TypeRetriever],
  compressor_types: list[TypeCompressor],
  search_type: TypeSearch,
  search_kwargs: dict,
  document_content_description: Union[str, None],
  metadata_field_info: Union[list, None]
) -> BaseRetriever:
  try:
    my_retrievers = []
    my_compressors = []

    #*----*----#
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

    #*----*----#

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
        tratransformers=my_compressors,
      )

      retriever_contextual_compression = ContextualCompressionRetriever(
        base_compressor=compressor_pipeline, base_retriever=retriever_base,
      )

      my_retrievers.append(retriever_contextual_compression)

    #*----*----#

    logger.info(f"[Retrievers] {retriever_types}")
    retriever_ensemble = EnsembleRetriever(
      retrievers=my_retrievers,
      weights=create_portion(my_retrievers),
    )

    return retriever_ensemble
  except Exception as e:
    logger.error(f"[Retrievers] Error:: {e}")
    raise

class BaseStore:
  def __init__(
    self,
    embeddings_provider: EmbeddingsProvider = "openai",
    embeddings_model: EmbeddingsModel = "text-embedding-ada-002",
    tool_name: str = "",
    tool_description: str = "",
    llm: Union[BaseChatModel, None] = None,
    retriever_types: list[TypeRetriever] = [
      'base', 'MultiQueryRetriever', # 'RePhraseQueryRetriever',
    ],
    configs: Dict = None,
    compressor_types: list[TypeCompressor] = [],
    search_type: TypeSearch = "mmr",
    search_kwargs: dict = {
      "k": 6,
    },
    document_content_description: Union[str, None] = None,
    metadata_field_info: Union[list, None] = None,
    **kwargs,
  ):
    self.configs = configs
    self.embeddings_provider = embeddings_provider
    self.embeddings_model = embeddings_model
    self.tool_name = tool_name
    self.tool_description = tool_description
    self.llm = llm
    self.retriever_types = retriever_types
    self.compressor_types = compressor_types
    self.search_type = search_type
    self.search_kwargs = search_kwargs
    self.document_content_description = document_content_description
    self.metadata_field_info = metadata_field_info

    self.embeddings = create_embeddings(embeddings_provider, embeddings_model)
    self.embeddings_size = get_embeddings_size(embeddings_model)
    self.vector_store: VectorStore = None
    self.retriever: BaseRetriever = None
    self.retriever_tool = None

  def create_retriever_and_tool(self, vector_store: VectorStore):
    self.retriever = create_retriever(
      llm=self.llm,
      vectorstore=vector_store,
      embeddings=self.embeddings,
      retriever_types=self.retriever_types,
      compressor_types=self.compressor_types,
      search_type=self.search_type,
      search_kwargs=self.search_kwargs,
      document_content_description=self.document_content_description,
      metadata_field_info=self.metadata_field_info
    )
    self.create_retriever_tool()

  def add_documents(self, docs: List[Document]):
    if self.vector_store is not None:
      for doc in tqdm(docs):
        self.vector_store.add_documents([doc])
    else:
      raise NotImplementedError("Vector store is not initialized")

  def invoke_retriever(self, query, **kwargs) -> List[Document]:
    if self.retriever is not None:
      result = self.retriever.invoke(query, **kwargs)
      return [Document(res.page_content) for res in result]
    else:
      raise NotImplementedError("Retriever is not initialized")

  def create_retriever_tool(self):
    if self.retriever is not None:
      self.retriever_tool = create_retriever_tool(
        retriever=self.retriever,
        name=self.tool_name,
        description=self.tool_description,
      )
    else:
      raise NotImplementedError("Retriever is not initialized")

class QdrantStore(BaseStore):
  def __init__(
    self,
    collection_name: str = "",
    distance: QdrantTypeDistance = "Cosine",
    prefer_grpc: bool = False,
    content_payload_key: str = "page_content",
    metadata_payload_key: str = "metadata",
    **kwargs
  ):
    super().__init__(**kwargs)
    self.distance = distance
    
    if not os.getenv("QDRANT_HOST") or not os.getenv("QDRANT_API_KEY"):
      logger.error("Please put QDRANT_HOST and QDRANT_API_KEY in .env file.")
      
    self.client = qdrant_client.QdrantClient(
      location=os.getenv("QDRANT_HOST"),
      api_key=os.getenv("QDRANT_API_KEY"),
      prefer_grpc=prefer_grpc,
    )
    self.collection_name = collection_name
    
    self.setup_collection()
    
    self.vector_store = Qdrant(
      client=self.client,
      collection_name=self.collection_name,
      embeddings=self.embeddings,
      content_payload_key=content_payload_key,
      metadata_payload_key=metadata_payload_key,
    )
    
    self.create_retriever_and_tool(self.vector_store)

  def setup_collection(self):
    client_collections = str(self.client.get_collections())
    if self.collection_name in client_collections:
      logger.info(f"[Qdrant] Collection `{self.collection_name}` found.")
    else:
      self.client.create_collection(
        collection_name=self.collection_name,
        vectors_config=models.VectorParams(
          size=self.embeddings_size,
          distance=self.distance,
        ),
      )
      logger.info(f"[Qdrant] Collection `{self.collection_name}` created.")

class MongoStore(BaseStore):
  def __init__(self, mongodb_atlas_cluster_uri: str, db_name: str, collection_name: str, index_name: str, dimensions: int, **kwargs):
    super().__init__(**kwargs)
    self.client = MongoClient(mongodb_atlas_cluster_uri)
    self.db_name = db_name
    self.collection_name = collection_name
    self.index_name = index_name
    self.dimensions = dimensions
    self.collection = self.client[db_name][collection_name]
    self.vector_store = MongoDBAtlasVectorSearch(
      collection=self.collection,
      index_name=self.index_name,
      embedding=self.embeddings,
      relevance_score_fn="cosine",
    )
    self.create_retriever_and_tool()

  def create_index(self):
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
      logger.success(f"[Mongo] Index '{self.index_name}' created.")
      return result
    except Exception as e:
      if "already exists" in str(e):
        logger.info(f"[Mongo] Index '{self.index_name}' already exists.")
      else:
        logger.error(f"[Mongo] Error creating index: {e}")
      return None
