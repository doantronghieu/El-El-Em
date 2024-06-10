import add_packages
import os
import yaml

from toolkit.langchain import (
  prompts, agents, stores, models, chains, text_embedding_models
)
from toolkit import sql

# *=============================================================================
# call from main
with open(f"{add_packages.APP_PATH}/my_configs/vtc.yaml", 'r') as file:
  configs = yaml.safe_load(file)

llm = models.chat_openai
embeddings = text_embedding_models.OpenAIEmbeddings()
vectorstore = stores.faiss.FAISS

#*------------------------------------------------------------------------------

examples_fewshot_tmp = dict(configs["sql"]["examples_questions_to_sql"]).values()
examples_questions_to_sql = [example for sublist in examples_fewshot_tmp for example in sublist]

proper_nouns = configs["sql"]["proper_nouns"]

my_sql_db = sql.MySQLDatabase()

cfg_sql_tool = configs["sql"]["tool"]

my_sql_chain = chains.MySqlChain(
	my_sql_db=my_sql_db,
	llm=llm,
	embeddings=embeddings,
	vectorstore=vectorstore,
	proper_nouns=proper_nouns,
	k_retriever_proper_nouns=4,
	examples_questions_to_sql=examples_questions_to_sql,
	k_few_shot_examples=5,
	sql_max_out_length=500,
	is_sql_get_all=True,
	is_debug=False,
	tool_name=cfg_sql_tool["name"],
	tool_description=cfg_sql_tool["description"],
	tool_metadata=cfg_sql_tool["metadata"],
	tool_tags=cfg_sql_tool["tags"],
)

tool_chain_sql = my_sql_chain.create_tool_chain_sql()

#*------------------------------------------------------------------------------

qdrant_txt_vtc_faq = stores.QdrantWrapper(
  qdrant_host=os.getenv("QDRANT_HOST"),
  qdrant_api_key=os.getenv("QDRANT_API_KEY"),
  configs=configs,
  **configs["vector_db"]["qdrant"]["vtc_faq"]
)

qdrant_txt_onli_faq = stores.QdrantWrapper(
  qdrant_host=os.getenv("QDRANT_HOST"),
  qdrant_api_key=os.getenv("QDRANT_API_KEY"),
  configs=configs,
  **configs["vector_db"]["qdrant"]["onli_faq"]
)

qdrant_lectures_content = stores.QdrantWrapper(
  qdrant_host=os.getenv("QDRANT_HOST"),
  qdrant_api_key=os.getenv("QDRANT_API_KEY"),
  configs=configs,
  **configs["vector_db"]["qdrant"]["lectures_content"],
)

#*==============================================================================
system_message_vtc = configs["prompts"]["system_message_vtc"]

prompt_onlinica = prompts.create_prompt_tool_calling_agent(
  system_message_vtc
)

tools = [
  # qdrant_lectures_content.retriever_tool,
  qdrant_txt_vtc_faq.retriever_tool,
	qdrant_txt_onli_faq.retriever_tool,
	tool_chain_sql,
]

system_message_custom = configs["prompts"]["system_message_vtc"]
prompt = prompts.create_prompt_tool_calling_agent(system_message_custom)

agent = agents.MyStatelessAgent(
	llm=llm,
	tools=tools,
	prompt=prompt,
	agent_type=configs["agents"]["type"],
	agent_verbose=False,
)