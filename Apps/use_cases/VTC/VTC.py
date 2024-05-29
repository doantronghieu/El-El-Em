import add_packages
import config
from pprint import pprint
import os
import yaml

from toolkit.langchain import (
  chat_models, prompts, agents, vectorstores
)


# *=============================================================================
# call from main
with open(f"{add_packages.APP_PATH}/my_configs/vtc.yaml", 'r') as file:
  configs = yaml.safe_load(file)

qdrant_lectures_content = vectorstores.QdrantWrapper(
  qdrant_host=os.getenv("QDRANT_HOST"),
  qdrant_api_key=os.getenv("QDRANT_API_KEY"),
  configs=configs,
  **configs["vector_db"]["qdrant"]["lectures_content"],
)

qdrant_courses_information = vectorstores.QdrantWrapper(
  qdrant_host=os.getenv("QDRANT_HOST"),
  qdrant_api_key=os.getenv("QDRANT_API_KEY"),
  configs=configs,
  **configs["vector_db"]["qdrant"]["courses_information"]
)

qdrant_faq = vectorstores.QdrantWrapper(
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

llm = chat_models.create_chat_model(configs)

agent = agents.MyStatelessAgent(
	llm=llm,
	tools=tools,
	prompt=prompt_onlinica,
	agent_type='tool_calling',
	agent_verbose=False,
)