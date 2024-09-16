import os, dotenv, yaml
from typing import Literal, Union, TypeAlias
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel

dotenv.load_dotenv()

MODEL_PROVIDER: TypeAlias = Literal["openai", "gemini", "anthropic", "groq", "cohere"]
MODEL_VERSION_OPENAI: TypeAlias = Literal[
  "gpt-3.5-turbo-0125", "gpt-4-turbo-preview", "gpt-4o-mini"
]
MODEL_VERSION_ANTHROPIC: TypeAlias = Literal[
  "claude-3-haiku-20240307", "claude-3-opus-20240229", "claude-3-sonnet-20240229",
]
MODEL_VERSION_GROQ: TypeAlias = Literal[
  "mixtral-8x7b-32768", "llama3-70b-8192",
]
MODEL_VERSION_COHERE: TypeAlias = Literal[
  "command", "command-r", "command-r-plus",
]

chat_openai = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125", streaming=True)

def create_llm(
  provider: MODEL_PROVIDER,
  version: Union[
    MODEL_VERSION_OPENAI, MODEL_VERSION_ANTHROPIC, MODEL_VERSION_GROQ,
    MODEL_VERSION_COHERE,
  ],
  temperature: float = 0,
  streaming: bool = True,
  **kwargs,
) -> BaseChatModel:
  model = None
  
  if provider == "openai":
    return ChatOpenAI(model=version, temperature=temperature, streaming=streaming)
  elif provider == "anthropic":
    return ChatAnthropic(model=version, temperature=temperature, streaming=streaming)
  elif provider == "groq":
    return ChatGroq(model=version, temperature=temperature, streaming=streaming)
  elif provider == "cohere":
    return ChatCohere(model=version, temperature=temperature, streaming=streaming)
  
  return model
