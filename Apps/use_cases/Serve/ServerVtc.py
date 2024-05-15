import add_packages
import os
import dotenv
import yaml

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from toolkit.langchain import (
    prompts, chat_models,
)

from use_cases.VTC import VTC

# *============================================================================

app = FastAPI(
  title="VTC Chatbot Server",
)

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

#*============================================================================

@app.post("/invoke-llm")
async def invoke_llm(
  msg: str
):
  ai_msg = VTC.agent.invoke_agent(msg)
  return ai_msg

@app.post("/stream-llm")
async def stream_llm(
  msg: str
):
  ai_msg = ""

  async for chunk in VTC.agent.agent_executor.astream(msg):
    ai_msg += chunk
    print(chunk.content, end="|", flush=True)

  return ai_msg

if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=8000)
