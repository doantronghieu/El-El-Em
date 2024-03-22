import add_packages
import config
from pprint import pprint
import os

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

qdrant_instance_lectures_content = vector_stores.QdrantWrapper(
    collection_name="vtc-lectures-content-2",
    qdrant_host=os.getenv("QDRANT_HOST"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    default_search_type="similarity",
    default_search_kwargs={"k": 10},
    retriever_tool_name="lectures_content",
    retriever_tool_description="Searches and returns content, knowledge from \
    the lecture scripts based on specialized keywords from user's question like \
    Typography, Lazada, Premiere, Unity ...",
)

qdrant_instance_courses_information = vector_stores.QdrantWrapper(
    collection_name="vtc-courses-information-2",
    qdrant_host=os.getenv("QDRANT_HOST"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    default_search_type="similarity",
    default_search_kwargs={"k": 20},
    retriever_tool_name="courses_information",
    retriever_tool_description="Searches and returns information about courses \
      of Onlinica like course name, course category, course link, course \
      description, total number of courses ...",
)

qdrant_instance_faq = vector_stores.QdrantWrapper(
    collection_name="vtc-faq",
    qdrant_host=os.getenv("QDRANT_HOST"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    default_search_type="similarity",
    default_search_kwargs={"k": 10},
    retriever_tool_name="frequently_asked_questions",
    retriever_tool_description="Searches and returns answer for frequently asked \
    questions about Onlinica information like accounts, fees, courses, payments, \
    certificates ...",
)

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

