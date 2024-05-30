import add_packages
from toolkit.langchain import models
import config
from pprint import pprint
import os
import yaml

from toolkit.langchain import (
  prompts, agents, stores
)

# *=============================================================================
# call from main
with open(f"{add_packages.APP_PATH}/my_configs/vtc.yaml", 'r') as file:
  configs = yaml.safe_load(file)

qdrant_lectures_content = stores.QdrantWrapper(
  qdrant_host=os.getenv("QDRANT_HOST"),
  qdrant_api_key=os.getenv("QDRANT_API_KEY"),
  configs=configs,
  **configs["vector_db"]["qdrant"]["lectures_content"],
)

qdrant_courses_information = stores.QdrantWrapper(
  qdrant_host=os.getenv("QDRANT_HOST"),
  qdrant_api_key=os.getenv("QDRANT_API_KEY"),
  configs=configs,
  **configs["vector_db"]["qdrant"]["courses_information"]
)

qdrant_faq = stores.QdrantWrapper(
  qdrant_host=os.getenv("QDRANT_HOST"),
  qdrant_api_key=os.getenv("QDRANT_API_KEY"),
  configs=configs,
  **configs["vector_db"]["qdrant"]["faq"]
)

# *=============================================================================
system_message_onlinica = configs["prompts"]["system_message_onlinica"]

prompt_onlinica = prompts.create_prompt_tool_calling_agent(
  system_message_onlinica
)

tools = [
  qdrant_lectures_content.retriever_tool,
  qdrant_faq.retriever_tool,
  qdrant_courses_information.retriever_tool,
]

llm = models.create_chat_model(configs)

agent = agents.MyStatelessAgent(
	llm=llm,
	tools=tools,
	prompt=prompt_onlinica,
	agent_type='tool_calling',
	agent_verbose=False,
)