import add_packages
import asyncio
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from toolkit.langchain import (
  chat_models, agent_tools, agents, prompts, runnables,
)
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

history = agents.ChatHistory(
	history_type='in_memory',
	user_id="admin",
	session_id="1",
)

my_agent = agents.MyAgent(
	llm=my_llm,
	tools=my_tools,
	prompt=my_prompt,
	history=history,
	agent_verbose=False, # True, False
	agent_type='tool_calling',
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

def stream_generator_chat_model(
  query: str,
  chat_model: runnables.Runnable,
):
  for token in chat_model.stream(query):
    yield(token.content)

@app.get("/stream-chat-model")
async def stream_chat_model(
  query: str = "Hello",
):
  return StreamingResponse(
    stream_generator_chat_model(query, chat_models.chat_openai),
    media_type='text/event-stream',
  )

#*------------------------------------------------------------------------------

def stream_generator_agent(
  query: str,
  agent: agents.MyAgent,
):
  return agent.astream_events_basic(query)

@app.get("/stream-agent")
async def stream_agent(
  query: str = "Hello",
):
  return StreamingResponse(
    stream_generator_agent(query=query, agent=my_agent),
    media_type='text/event-stream',
  )

@app.get("/invoke-agent")
async def invoke_agent(
  query: str = "Hello",
):
  return Response(
    content=await my_agent.invoke_agent(query),
  )

@app.get("/agent-chat-history")
async def get_agent_chat_history():
  result = await my_agent.history._get_chat_history()
  return result
  # return JSONResponse(result_json)


@app.delete("/agent-chat-history")
async def clear_agent_chat_history():
  return await my_agent.history.clear_chat_history()
#*==============================================================================
# uvicorn server:app --host=0.0.0.0 --port=8000