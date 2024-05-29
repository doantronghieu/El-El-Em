import config
import os, dotenv, yaml
from loguru import logger
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel

dotenv.load_dotenv()

# with open("../my_configs/vtc.yaml") as file:
#   configs = yaml.safe_load(file)

chat_openai = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125", streaming=True)

chat_anthropic = ChatAnthropic(
  temperature=0, model_name="claude-3-haiku-20240307",
)

chat_groq_mixtral = ChatGroq(
  temperature=0, model_name="mixtral-8x7b-32768",
)

chat_cohere = ChatCohere(
  model="command-r-plus", # "command", "command-r", "command-r-plus",
  temperature=0
)

def create_chat_model(config: dict) -> BaseChatModel:
  model_option = config["model"]["option"]
  model_version = config['model'][model_option]
  temperature = config["model"]["temperature"]
  
  logger.info(f"Model: {model_option}, {model_version}")
  
  if model_option == 'openai':
      if model_version:
          return ChatOpenAI(
              temperature=temperature, model=model_version, streaming=True,
          )
      else:
          raise ValueError("OpenAI model name is missing in config.")
  elif model_option == 'anthropic':
      if model_version:
          return ChatAnthropic(
              temperature=temperature, model_name=model_version, streaming=True,
          )
      else:
          raise ValueError("Anthropic model name is missing in config.")
  else:
      raise ValueError(
          "Invalid model option in config. Supported options are 'openai', 'anthropic'.")

