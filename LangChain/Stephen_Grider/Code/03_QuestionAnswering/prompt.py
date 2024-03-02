from dotenv import load_dotenv
import langchain
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from redundant_filter_retriever import RedundantFilterRetriever

langchain.debug = True
load_dotenv()

chat = ChatOpenAI()
embeddings = OpenAIEmbeddings()
db = Chroma(persist_directory='emb', embedding_function=embeddings)
retriever = RedundantFilterRetriever(embeddings=embeddings, chroma=db)
chain = RetrievalQA.from_chain_type(llm=chat, retriever=retriever,
  chain_type='stuff' # map_reduce 
) 

result = chain.run('What is an interesting fact about the English language?')
print(result)