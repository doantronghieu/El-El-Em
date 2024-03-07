import add_packages
import config
from pprint import pprint
import os

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
    collection_name="vtc-lectures-content",
    qdrant_host=os.getenv("QDRANT_HOST"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    default_search_type="similarity",
    default_search_kwargs={"k": 10},
    retriever_tool_name="lectures_content",
    retriever_tool_description="Searches and returns content, knowledge from \
    the lecture scripts based on specialized keywords from user's question like \
    Typography, Lazada, Premiere, Unity ...",
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

qdrant_instance_courses_information = vector_stores.QdrantWrapper(
    collection_name="vtc-courses-information",
    qdrant_host=os.getenv("QDRANT_HOST"),
    qdrant_api_key=os.getenv("QDRANT_API_KEY"),
    default_search_type="similarity",
    default_search_kwargs={"k": 10},
    retriever_tool_name="courses_information",
    retriever_tool_description="Searches and returns information about courses \
      of Onlinica like course name, course category, course link, course \
      description, total number of courses ...",
)

onlinica_system_message = """\
You are a consultant for an online learning platform called Onlinica.

You have the following qualities:
- Helpful
- Extremely dedicated and hardworking
- Professionalism, respect, sincerity and honesty
- Standard thinking
- Excellent communication, negotiation and complaint handling skills
- Excellent sales skills
- Deep understanding of products/services. Strong knowledge of the industry
- Optimistic and positive spirit. Ability to create a positive customer experience
- Sensitive to customers' requests and desires

You will help users answer questions about the courses on the platform. The language in which you respond will be the same as the user's language.

Questions users might ask and how to answer:
- Course basics: You SHOULD list ALL available courses (and their information) that are RELEVANT to the user's question.
- Content and knowledge of the lecture: These questions will often contain specialized keywords such as Typography, Lazada, Premiere, Unity,... You will synthesize information from the scripts of the lectures that contain keywords that major and give detailed answers to users. You should suggest courses (course name with course link) related to that specialized keyword to users.
- Frequently asked questions\
"""
onlinica_prompt = prompts.create_openai_tools_agent_custom_prompt(
    onlinica_system_message)

tools = [
    qdrant_instance_lectures_content.retriever_tool,
    qdrant_instance_faq.retriever_tool,
    qdrant_instance_courses_information.retriever_tool,
]

agent_prompt = prompts.openai_tools_agent_prompt

agent = agents.create_openai_tools_agent(
    llm=chat_models.chat_openai,
    tools=tools,
    prompt=onlinica_prompt,
)
agent_executor = agents.AgentExecutor(agent=agent, tools=tools, verbose=True)
