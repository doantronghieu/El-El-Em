from langserve import add_routes
from langchain.pydantic_v1 import BaseModel, Field
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from fastapi import FastAPI
from typing import List
from config import set_environment
set_environment()


# Load Retriever
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

# Create tools
retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="langsmith_search",
    description=("Search for information about LangSmith. For any questions about "
                 "LangSmith, you must use this tool!"),
)
search_tool = TavilySearchResults()
tools = [retriever_tool, search_tool]

# Create Agent
prompt = hub.pull("hwchase17/openai-functions-agent")
llm = ChatOpenAI(temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# App definition
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API server using LangChain's Runnable interfaces",
)

# Adding chain route

class Input(BaseModel):
    input: str
    chat_history: List[BaseMessage] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "location"}},
    )

class Output(BaseModel):
  output: str
  
add_routes(
  app=app,
  runnable=agent_executor.with_types(input_type=Input, output_type=Output),
  path="/agent",
)

if __name__ == "__main__":
  import uvicorn
  
  uvicorn.run(app, host="localhost", port=8000)
  
# python serve.py