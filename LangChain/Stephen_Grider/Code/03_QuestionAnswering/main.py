from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(separator='\n', chunk_size=200, 
  chunk_overlap=0)

loader = TextLoader('facts.txt')
docs = loader.load_and_split(text_splitter)

db = Chroma.from_documents(docs, embedding=embeddings, persist_directory='emb')

result = Chroma.similarity_search('What is an interesting fact about the English language?')
