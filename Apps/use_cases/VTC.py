import add_packages
import config
from pprint import pprint
import os, json

from my_prompts import (
    prompts_onlinica
)

from my_langchain import (
    document_loaders, text_splitters, text_embedding_models, vector_stores,
    chat_models, prompts, utils, output_parsers, agents, documents
)

from langchain_core.prompts import (
    ChatPromptTemplate, PromptTemplate, MessagesPlaceholder,
    SystemMessagePromptTemplate, AIMessagePromptTemplate,
    HumanMessagePromptTemplate
)

#*=============================================================================
# call from main
with open("./my_configs/vtc.json", "r") as json_file:
    vtc_data = json.load(json_file)

qdrant_instance_lectures_content = vector_stores.QdrantWrapper(
    collection_name=os.getenv("DB_QDRANT_COLLECTION_LECTURES_CONTENT"),
    qdrant_host=os.getenv("QDRANT_HOST"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    **vtc_data["vector_db"]["qdrant_instance_lectures_content"]
)

qdrant_instance_courses_information = vector_stores.QdrantWrapper(
    collection_name=os.getenv("DB_QDRANT_COLLECTION_COURSES_INFORMATION"),
    qdrant_host=os.getenv("QDRANT_HOST"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    **vtc_data["vector_db"]["qdrant_instance_courses_information"]
)

qdrant_instance_faq = vector_stores.QdrantWrapper(
    collection_name=os.getenv("DB_QDRANT_COLLECTION_FAQ"),
    qdrant_host=os.getenv("QDRANT_HOST"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    **vtc_data["vector_db"]["qdrant_instance_faq"]
)

#*=============================================================================
system_message_onlinica = prompts_onlinica.system_message_onlinica

prompt_onlinica = prompts.create_prompt_custom_agent_openai_tools(
    system_message_onlinica)

tools = [
    qdrant_instance_lectures_content.retriever_tool,
    qdrant_instance_faq.retriever_tool,
    qdrant_instance_courses_information.retriever_tool,
]

llm = chat_models.chat_openai
agent = agents.MyAgent(prompt=prompt_onlinica, tools=tools,
                       agent_type="openai_tools", llm=llm)

