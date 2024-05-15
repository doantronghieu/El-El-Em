import add_packages
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

#*==============================================================================

def stream_generator_langchain_agent(
  query: str,
  agent: agents.MyAgent,
):
  return agent.astream_events_basic(query)

@app.get("/stream-vtc-agent")
async def stream_vtc_agent(
  query: str = "Hello",
):
  return StreamingResponse(
    stream_generator_langchain_agent(query=query, agent=VTC.agent),
    media_type='text/event-stream',
  )
#*==============================================================================

if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8000)