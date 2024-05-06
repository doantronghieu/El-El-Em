from langchain_core.documents import Document
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader

def format_docs(docs: list[Document]):
	return "\n\n".join(doc.page_content for doc in docs)