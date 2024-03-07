from langchain_core.prompts import (
    ChatPromptTemplate, PromptTemplate, MessagesPlaceholder,
    SystemMessagePromptTemplate, AIMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain_core.messages import (
  SystemMessage, AIMessage, HumanMessage,
)
from langchain import hub
# *-----------------------------------------------------------------------------

prompt_template = PromptTemplate

# *-----------------------------------------------------------------------------

general_template = """\
The following is a friendly conversation between a human and an AI. \
The AI is talkative and provides lots of specific details from its context. \
If the AI does not know the answer to a question, it truthfully say it does not \
know.

Current conversation:
{history}
Human: {input}
AI Assistant:"""
general_prompt = prompt_template(
    input_variables=["history", "input"], template=general_template,
)

# *-----------------------------------------------------------------------------
#######
# RAG #
#######

rag_template = """\
You are an assistant for question-answering tasks. Use the following pieces of \
retrieved context to answer the question at the end. If you don't know the \
answer, just say that you don't know, don't try to make up an answer. Use \
three sentences maximum and keep the answer as concise as possible.

Context: {context} 

Question: {question} 
Helpful Answer:\
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)
# *---
contextualize_q_system_prompt = """\
Given a chat history and the lastest user question which might reference context \
in the chat history, formulate a standalone question which can be understood \
without the chat history. Do NOT answer the question, just reformulate it if \
needed and otherwise return as is.\
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
  ("system", contextualize_q_system_prompt),
  MessagesPlaceholder(variable_name="chat_history"),
  ("human", "{question}"),
])

qa_system_prompt = """\
You are an assistant for question-answering tasks. Use the following pieces of \
retrieved context to answer the question at the end. If you don't know the \
answer, just say that you don't know, don't try to make up an answer. Use \
three sentences maximum and keep the answer as concise as possible.

{context}
"""

qa_prompt = ChatPromptTemplate.from_messages([
  ("system", qa_system_prompt),
  MessagesPlaceholder(variable_name="chat_history"),
  ("human", "{question}"),
])

# *-----------------------------------------------------------------------------

#########
# AGENT #
#########

# hwchase17/openai-tools-agent
# openai_tools_agent_prompt = hub.pull("hwchase17/openai-tools-agent")
openai_tools_agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

def create_openai_tools_agent_custom_prompt(system_message: str):
  openai_tools_agent_custom_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_message),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
  )
  
  return openai_tools_agent_custom_prompt
