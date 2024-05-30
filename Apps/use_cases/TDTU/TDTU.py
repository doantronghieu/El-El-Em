import add_packages
from toolkit.langchain import models
from pprint import pprint
import os
import yaml

from toolkit.langchain import (
  prompts, agents, stores
)


# *=============================================================================
# call from main
with open(f"{add_packages.APP_PATH}/my_configs/tdtu.yaml", 'r') as file:
  configs = yaml.safe_load(file)

qdrant_csv_personnel = stores.QdrantWrapper(
	qdrant_host=os.getenv("QDRANT_HOST"),
	qdrant_api_key=os.getenv("QDRANT_API_KEY"),
	configs=configs,
	**configs["vector_db"]["qdrant"]["personnel"]
)

qdrant_csv_admission = stores.QdrantWrapper(
	qdrant_host=os.getenv("QDRANT_HOST"),
	qdrant_api_key=os.getenv("QDRANT_API_KEY"),
	configs=configs,
	**configs["vector_db"]["qdrant"]["university_admission_program"]
)

qdrant_txt_info = stores.QdrantWrapper(
  qdrant_host=os.getenv("QDRANT_HOST"),
  qdrant_api_key=os.getenv("QDRANT_API_KEY"),
  configs=configs,
  **configs["vector_db"]["qdrant"]["general_information"]
)

# *=============================================================================
llm = models.chat_openai

tools = [
	# agent_tools.TavilySearchResults(max_results=3),
	qdrant_csv_admission.retriever_tool,
	qdrant_csv_personnel.retriever_tool,
	qdrant_txt_info.retriever_tool,
]

system_message_custom = configs["prompts"]["system_message_tdtu"]
prompt = prompts.create_prompt_tool_calling_agent(system_message_custom)

agent = agents.MyStatelessAgent(
	llm=llm,
	tools=tools,
	prompt=prompt,
	agent_type=configs["agents"]["type"],
	agent_verbose=False,
)