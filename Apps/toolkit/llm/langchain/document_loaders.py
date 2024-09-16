from typing import Any, Literal, TypeAlias
from loguru import logger

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import (
  UnstructuredExcelLoader, UnstructuredPowerPointLoader, UnstructuredEmailLoader
)
from langchain_community.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
from langchain_community.document_loaders.html import UnstructuredHTMLLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.document_loaders.youtube import YoutubeLoader

TypeDocumentSrc: TypeAlias = Literal[
	"csv", "excel", "power_point", "word", "email",
	"web", "html", "json", "markdown", "pdf", "youtube", 
]

class MyDocumentLoaders:
	def __init__(
		self,
		src: Any,
		src_type: TypeDocumentSrc, 
		**kwargs,
  ) -> None:
		self.src = src
		self.src_type = src_type
  
		self.loader: BaseLoader = None
		self.data = None
  
		if src_type == "csv":
			self.loader = CSVLoader(
				file_path=self.src,
				csv_args={
					"delimiter": ",",
					"quotechar": '"',
					"fieldnames": ["a", "b", "c"],
				},
				source_column="Source",
			)
		elif src_type == "excel":
			self.loader = UnstructuredExcelLoader(file_path=self.src, mode="elements")
		elif src_type == "power_point":
			self.loader = UnstructuredPowerPointLoader(file_path=self.src, mode="elements")
		elif src_type == "word":
			self.loader = UnstructuredWordDocumentLoader(file_path=self.src, mode="elements")
		elif src_type == "email":
			self.loader = UnstructuredEmailLoader(file_path=self.src, mode="elements")
		elif src_type == "web":
			self.loader = WebBaseLoader(web_path=self.src)
		elif src_type == "html":
			self.loader = UnstructuredHTMLLoader(file_path=self.src)
		elif src_type == "json":
			...
		elif src_type == "markdown":
			self.loader = UnstructuredMarkdownLoader(file_path=self.src, mode="elements")
		elif src_type == "pdf":
			self.loader = UnstructuredPDFLoader(file_path=self.src)
		elif src_type == "youtube":
			self.loader = YoutubeLoader.from_youtube_url(
				youtube_url=self.src,
				add_video_info=True,
				language=["en",],
				translation="en",
			)
   
		logger.info(f"Document Type: {src_type}")
  
	async def load_data(
		self,
	):
		self.data = await self.loader.aload()
		return self.data

def format_docs(docs: list[Document]):
	return "\n\n".join(doc.page_content for doc in docs)