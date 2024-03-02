import logging
import pathlib
from typing import Any

from langchain.document_loaders import (PyPDFLoader, TextLoader,
                                        UnstructuredEPubLoader,
                                        UnstructuredWordDocumentLoader)
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document

################################################################################


def init_memory():
    """
    Initialize the memory for contextual conversation.

    Caching memory so it won't be deleted everytime we restart the server.
    """
    return ConversationBufferMemory(memory_key='chat_history',
                                    return_messages=True, output_key='answer')


MEMORY = init_memory()
################################################################################


class EpubReader(UnstructuredEPubLoader):
    def __init__(self, file_path: str | list[str], **unstructured_kwargs: Any):
        super().__init__(file_path, **unstructured_kwargs, mode="elements",
                         strategy="fast")


class DocumentLoaderException(Exception):
    pass


class DocumentLoader(object):
    """
    Loads in document with a supported extensions.
    """

    supported_extensions = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.epub': EpubReader,
        '.docx': UnstructuredWordDocumentLoader,
        '.doc': UnstructuredWordDocumentLoader
    }

def load_document(temp_filepath: str) -> list[Document]:
  """
  Load a file and return it as a list of documents.
  """
  
  ext = pathlib.Path(temp_filepath).suffix
  loader = DocumentLoader.supported_extensions.get(ext)
  
  if (not loader):
    raise DocumentLoaderException(
      f'Invalid extention type `{ext}`, cannot load this type of file.'
    )
    
  loaded = loader(temp_filepath)
  docs = loaded.load()
  
  logging.info(docs)
  
  return docs
