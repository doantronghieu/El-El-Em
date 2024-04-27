from typing import Union, Sequence
import streamlit as st
import uuid
import add_packages

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable

from my_langchain import chat_models, agent_tools, prompts, agents

from my_streamlit import utils
from my_streamlit.utils import CHAT_ROLE, MSG_ITEM

#*==============================================================================

st.set_page_config(
  layout="wide",
)

STATES = {
  "MESSAGES": {
    "INITIAL_VALUE": [
      prompts.AIMessage(content="Your are a helpful assistant.")
    ],
  },
}


utils.initialize_session_state(STATES)



#*==============================================================================

@st.cache_resource
def create_agent(
  _llm: BaseLanguageModel, 
  _tools: Sequence[BaseTool], 
  _prompt: ChatPromptTemplate,
  _agent_type: str,
) -> Runnable:
  agent = agents.MyAgent(
    llm=_llm, tools=_tools, prompt=_prompt, agent_type=_agent_type
  )
  return agent

def create_callbacks() -> list:
  st_callback = utils.StreamlitCallbackHandler(st.container())
  callbacks = [st_callback]
  return callbacks

def generate_response(
  input, 
  agent: agents.MyAgent,
):
  
  response = agent.invoke_agent(
    input_message=input, 
    callbacks=create_callbacks()
  )

  return response

def process_chat(
  prompt: str, 
  agent: agents.MyAgent,
):
  st.chat_message(CHAT_ROLE.user).markdown(prompt)
  stream = generate_response(prompt, agent)
  response = st.chat_message(CHAT_ROLE.assistant).write(stream)
  return response

#*==============================================================================

llm = chat_models.chat_openai

tool_search = agent_tools.TavilySearchResults(max_results=3)
tools = [
  tool_search,
]

prompt = prompts.create_custom_prompt_tool_calling_agent()

agent: agents.MyAgent = create_agent(
  _llm=llm, 
  _tools=tools, 
  _prompt=prompt, 
  _agent_type="tool_calling"
)

#*==============================================================================

for msg in agent.chat_history:
  msg: Union[prompts.AIMessage, prompts.HumanMessage]
  st.chat_message(msg.type).markdown(msg.content)

prompt = st.chat_input("Say something")
  
if prompt:
  process_chat(prompt=prompt, agent=agent)
  