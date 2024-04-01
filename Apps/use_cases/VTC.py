import add_packages
import config
from pprint import pprint
import os
import yaml

from my_prompts import (
    prompts_onlinica
)

from my_langchain import (
    document_loaders, text_splitters, text_embedding_models, chat_models, prompts, utils, output_parsers, agents, documents, vectorstores
)

from langchain_core.prompts import (
    ChatPromptTemplate, PromptTemplate, MessagesPlaceholder,
    SystemMessagePromptTemplate, AIMessagePromptTemplate,
    HumanMessagePromptTemplate
)

# *=============================================================================
# call from main
with open("./my_configs/vtc.yaml", 'r') as file:
    configs_vtc = yaml.safe_load(file)

qdrant_lectures_content = vectorstores.QdrantWrapper(
    qdrant_host=os.getenv("QDRANT_HOST"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    configs=configs_vtc,
    **configs_vtc["vector_db"]["qdrant"]["lectures_content"],
)

qdrant_courses_information = vectorstores.QdrantWrapper(
    qdrant_host=os.getenv("QDRANT_HOST"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    configs=configs_vtc,
    **configs_vtc["vector_db"]["qdrant"]["courses_information"]
)

qdrant_faq = vectorstores.QdrantWrapper(
    qdrant_host=os.getenv("QDRANT_HOST"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    configs=configs_vtc,
    **configs_vtc["vector_db"]["qdrant"]["faq"]
)

# *=============================================================================
system_message_onlinica = configs_vtc["prompts"]["system_message_onlinica"]

prompt_onlinica = prompts.create_prompt_custom_agent_openai_tools(
    system_message_onlinica)

tools = [
    qdrant_lectures_content.retriever_tool,
    qdrant_faq.retriever_tool,
    qdrant_courses_information.retriever_tool,
]

llm = chat_models.create_chat_model(configs_vtc)
agent = agents.MyAgent(
    prompt=prompt_onlinica, tools=tools,
    agent_type=configs_vtc["agents"]["agent_type_onlinica"], 
    llm=llm
)
