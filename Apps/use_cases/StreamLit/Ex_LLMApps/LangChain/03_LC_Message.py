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
    "INITIAL_VALUE": [],
  },
  "PROMPT_EXAMPLE": {
    "INITIAL_VALUE": None,
  },
  "SELECTED_CHAT": {
    "INITIAL_VALUE": None,
  },
  "BTN_NEW_CHAT": {
    "INITIAL_VALUE": "widget",
  },
  "BTN_CLEAR_CHAT_HISTORY": {
    "INITIAL_VALUE": "widget",
  },
  "BTN_CLEAR_CHAT": {
    "INITIAL_VALUE": "widget",
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
  input: str, 
  agent: agents.MyAgent,
):
  response = agent.invoke_agent(
    input_message=input, 
    callbacks=create_callbacks()
  )
  return response

def process_on_user_input(
  prompt: str, 
  agent: agents.MyAgent,
):
  st.chat_message(CHAT_ROLE.user).markdown(prompt)
  stream = generate_response(prompt, agent)
  st.chat_message(CHAT_ROLE.assistant).write(stream)
  
  
def render_chat_messages_on_rerun(
  agent: agents.MyAgent,
):
  for msg in agent.chat_history:
    msg: Union[prompts.AIMessage, prompts.HumanMessage]
    st.chat_message(msg.type).markdown(msg.content)

def on_click_btn_clear_chat_history(
  agent: agents.MyAgent,
):
  agent.clear_chat_history()

def on_click_btn_new_chat(
  
):
  pass

def on_click_btn_clear_chat(
  
):
  pass
#*==============================================================================

llm = chat_models.chat_openai

tool_search = agent_tools.TavilySearchResults(max_results=3)
tools = [
  tool_search,
]

prompt = prompts.create_prompt_tool_calling_agent()

agent: agents.MyAgent = create_agent(
  _llm=llm, 
  _tools=tools, 
  _prompt=prompt, 
  _agent_type="tool_calling"
)

#*==============================================================================

render_chat_messages_on_rerun(agent=agent)

with st.sidebar:
  prompt_example = st.selectbox(
    label="Examples",
    label_visibility="collapsed",
    placeholder="Choose an example",
    options=[
      None,
      "Hello",
      "My name is Bob",
      "What is my name?",
      "Tell me a super long story about a dog",
      "What is the question I just asked you?",
    ],
    key=STATES["PROMPT_EXAMPLE"]["KEY"],
  )  
  
  #*----------------------------------------------------------------------------
  
  cols_chat = st.columns([0.15, 0.01, 0.84])
  
  btn_new_chat = cols_chat[0].button(
    label="✏️", 
    key=STATES["BTN_NEW_CHAT"]["KEY"],
    on_click=on_click_btn_new_chat, 
    kwargs=dict()
  )
  
  selected_chat = cols_chat[2].selectbox(
    label="Chat",
    label_visibility="collapsed",
    placeholder="Choose a Chat",
    options=[
      None,
      "Dummy Chat 1",
      "Dummy Chat 2",
    ],
    key=STATES["SELECTED_CHAT"]["KEY"],
  )
  
  #*----------------------------------------------------------------------------
  
  cols_clear = st.columns([0.5, 0.5])
  
  btn_clear_chat_history = cols_clear[0].button(
    label="Clear History", 
    key=STATES["BTN_CLEAR_CHAT_HISTORY"]["KEY"],
    on_click=on_click_btn_clear_chat_history, 
    kwargs=dict(agent=agent)
  )
  
  btn_clear_chat = cols_clear[1].button(
    label="Clear Chat", 
    key=STATES["BTN_CLEAR_CHAT"]["KEY"],
    on_click=on_click_btn_clear_chat, 
    kwargs=dict()
  )

prompt: Union[str, None]

prompt = st.chat_input("Say something")

if prompt_example:
  prompt = prompt_example
  del st.session_state[STATES["PROMPT_EXAMPLE"]["KEY"]]
  
if prompt:
  process_on_user_input(prompt=prompt, agent=agent)

# st.write(st.session_state)