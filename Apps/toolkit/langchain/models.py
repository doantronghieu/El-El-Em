import os, dotenv, yaml
from typing import Literal, Union, TypeAlias
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel

dotenv.load_dotenv()

MODEL_PROVIDER: TypeAlias = Literal["openai", "gemini", "anthropic"]
MODEL_VERSION_OPENAI: TypeAlias = Literal[
  "gpt-3.5-turbo-0125", "gpt-4-turbo-preview"
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

chat_anthropic = ChatAnthropic(
  temperature=0, model_name="claude-3-haiku-20240307",
)

chat_groq_mixtral = ChatGroq(
  temperature=0, model_name="mixtral-8x7b-32768",
)

chat_groq_llama3 = ChatGroq(
  temperature=0, model_name="llama3-70b-8192",
)

chat_cohere = ChatCohere(
  model="command-r-plus", # "command", "command-r", "command-r-plus",
  temperature=0
)

def create_llm(
  model_provider: MODEL_PROVIDER,
  model_version: Union[
    MODEL_VERSION_OPENAI, MODEL_VERSION_ANTHROPIC, MODEL_VERSION_GROQ,
    MODEL_VERSION_COHERE,
  ],
  temperature: float = 0,
  streaming: bool = True,
  **kwargs,
) -> BaseChatModel:
  pass