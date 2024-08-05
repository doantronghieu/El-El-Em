from langchain_core.prompts import (
  ChatPromptTemplate, PromptTemplate, MessagesPlaceholder,
  SystemMessagePromptTemplate, AIMessagePromptTemplate,
  HumanMessagePromptTemplate,
  FewShotPromptTemplate,
)
from langchain_core.messages import (
  SystemMessage, AIMessage, HumanMessage, ToolMessage, BaseMessage, 
  FunctionMessage, AnyMessage
)
from langchain.schema import ChatMessage
from langchain import hub

from langchain_core.example_selectors import SemanticSimilarityExampleSelector

from langchain_community.agent_toolkits.sql.prompt import SQL_FUNCTIONS_SUFFIX
from langchain.chains.sql_database.prompt import SQL_PROMPTS
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

# prompt = hub.pull("hwchase17/openai-tools-agent")
def create_prompt_tool_calling_agent(
  system_message: str = "You are a helpful assistant"
):
  prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad"),
  ])

  return prompt


# prompt = hub.pull("hwchase17/xml-agent-convo")
def create_prompt_custom_agent_xml_tools(
    system_message: str = "You are a helpful assistant. Help the user answer any questions."
):
  prompt_custom_agent_xml_tools = ChatPromptTemplate(
    input_variables=['agent_scratchpad', 'input', 'tools'],
    partial_variables={'chat_history': ''},
    messages=[
      HumanMessagePromptTemplate(
        prompt=PromptTemplate(
          input_variables=['agent_scratchpad',
                            'chat_history', 'input', 'tools'],
          template=system_message + "\n\nYou have access to the following tools:\n\n{tools}\n\nIn order to use a tool, you can use <tool></tool> and <tool_input></tool_input> tags. You will then get back a response in the form <observation></observation>\nFor example, if you have a tool called 'search' that could run a google search, in order to search for the weather in SF you would respond:\n\n<tool>search</tool><tool_input>weather in SF</tool_input>\n<observation>64 degrees</observation>\n\nWhen you are done, respond with a final answer between <final_answer></final_answer>. For example:\n\n<final_answer>The weather in SF is 64 degrees</final_answer>\n\nBegin!\n\nPrevious Conversation:\n{chat_history}\n\nQuestion: {input}\n{agent_scratchpad}"
        )
      )
    ]
  )

  return prompt_custom_agent_xml_tools

