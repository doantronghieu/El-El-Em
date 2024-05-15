import add_packages
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from toolkit.langchain import chat_models, agent_tools, agents, prompts, runnables

#*==============================================================================

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
  expose_headers=["*"],
)

@app.get("/")
async def redirect_root_to_docs():
  return RedirectResponse("/docs")

#*------------------------------------------------------------------------------
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
tools = [TavilySearchResults(max_results=1)]
prompt = ChatPromptTemplate.from_messages(
  [
    (
      "system",
      "You are a helpful assistant. Make sure to use the tavily_search_results_json tool for information.",
    ),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
  ]
)

# Construct the Tools agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#*==============================================================================
async def fake_data_streamer():
  for i in range(10):
      yield b'some fake data\n\n'
      await asyncio.sleep(0.5)

@app.get('/fake-stream')
async def fake_stream():
    return StreamingResponse(fake_data_streamer(), media_type='text/event-stream')
  
#*------------------------------------------------------------------------------
def stream_generator_langchain_chat_model(
  query: str,
  chat_model: runnables.Runnable,
):
  for token in chat_model.stream(query):
    yield(token.content)

@app.get("/stream-langchain-chat-model")
async def stream_langchain_chat_model(
  query: str = "Hello",
):
  return StreamingResponse(
    stream_generator_langchain_chat_model(query, chat_models.chat_openai),
    media_type='text/event-stream',
  )

#*------------------------------------------------------------------------------
def stream_generator_langchain_agent(
  query: str,
  agent: runnables.Runnable,
):
  for token in agent.stream({"input": query, "chat_history": []}):
    yield(token)

@app.get("/stream-langchain-agent")
async def stream_langchain_agent(
  query: str = "Hello",
):
  return StreamingResponse(
    stream_generator_langchain_agent(query, agent_executor),
    media_type='text/event-stream',
  )
#*==============================================================================

if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8000)