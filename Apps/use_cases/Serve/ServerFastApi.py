import add_packages
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from toolkit.langchain import (
  chat_models, agent_tools, agents, prompts, runnables,
)
from use_cases.VTC import VTC
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

my_llm = chat_models.chat_openai
my_prompt = prompts.create_prompt_tool_calling_agent()
my_tools = [
	agent_tools.TavilySearchResults(max_results=3)
]

my_agent = agents.MyAgent(
  llm=my_llm, tools=my_tools, prompt=my_prompt, agent_verbose=False
)

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
  agent: agents.MyAgent,
):
  return agent.astream_events_basic(query)

@app.get("/stream-langchain-agent")
async def stream_langchain_agent(
  query: str = "Hello",
):
  return StreamingResponse(
    stream_generator_langchain_agent(query=query, agent=my_agent),
    media_type='text/event-stream',
  )

@app.get("/stream-vtc-agent")
async def stream_vtc_agent(
  query: str = "Hello",
):
  return StreamingResponse(
    stream_generator_langchain_agent(query=query, agent=VTC.agent),
    media_type='text/event-stream',
  )
#*==============================================================================
# uvicorn ServerFastApi:app --host=0.0.0.0 --port=8000