# Server deploys OpenAI chat model, Anthropic chat model, and chain for joke
# about topic.
import add_packages
import os
import dotenv
import yaml

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from langserve import add_routes

from toolkit.langchain import (
    prompts, chat_models,
)

# *============================================================================

dotenv.load_dotenv()
with open("./config.yaml") as f:
  config = yaml.safe_load(f)

# *============================================================================

app = FastAPI(
    title="LangChain Server",
    description="A simple API Server using LangChain's Runnable Interfaces",
)

# Set CORS headers when calling endpoint from the browser using FastAPI's
# built-in middleware.
# Set all CORS enabled origins
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
# *============================================================================

add_routes(
    app=app,
    runnable=chat_models.chat_openai,
    path="/openai"
)

add_routes(
    app=app,
    runnable=chat_models.chat_anthropic,
    path="/anthropic"
)

prompt = prompts.ChatPromptTemplate.from_template(
    "tell me a joke about {topic}"
)
chain = prompt | chat_models.chat_anthropic
add_routes(
    app=app,
    runnable=chain,
    path="/joke"
)

if __name__ == "__main__":
  import uvicorn

  uvicorn.run(app, host="0.0.0.0", port=config["port"])
