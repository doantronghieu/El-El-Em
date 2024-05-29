import add_packages
from my_configs import constants
import logging
from typing import Literal, Union
import typing_inspect

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import (
  ContextualCompressionRetriever, EnsembleRetriever, RePhraseQueryRetriever
)
from langchain.retrievers.bm25 import BM25Retriever
from langchain.retrievers.document_compressors import (
  LLMChainExtractor, LLMChainFilter, EmbeddingsFilter, DocumentCompressorPipeline
)
from langchain_community.document_transformers.embeddings_redundant_filter import EmbeddingsRedundantFilter
from langchain_cohere import CohereRerank, CohereEmbeddings

logging.basicConfig()
logging.getLogger("langchain.retrievers.re_phraser").setLevel(logging.INFO)

from loguru import logger
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings

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
    'base', 'MultiQueryRetriever', 'CohereRerank'
  ],
  compressor_types: list[Literal[
    'EmbeddingsRedundantFilter', 'EmbeddingsFilter', 'LLMChainFilter', 
    'LLMChainExtractor',
  ]] = [],
  search_type: Literal['mmr', 'similarity'] = "mmr",
  search_kwargs: dict = {
    "k": 10,
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
