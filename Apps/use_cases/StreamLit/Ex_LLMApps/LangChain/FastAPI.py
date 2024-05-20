from typing import Union, Sequence
import asyncio
import uuid
import os
import time

from regex import P
import add_packages

import streamlit as st
from streamlit_feedback import streamlit_feedback

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable

from toolkit.langchain import (
  chat_models, agent_tools, prompts, agents, smiths, runnables, memories
)
from use_cases.Serve import client
from toolkit.streamlit import utils
from toolkit.streamlit.utils import CHAT_ROLE, MSG_ITEM

#*==============================================================================

st.set_page_config(
  layout="wide",
)

STATES = {
  "MESSAGES": {
    "INITIAL_VALUE": [],
  },
  "CONTAINER_PLACEHOLDER": {
    "INITIAL_VALUE": None,
  },
  "LAST_RUN": {
    "INITIAL_VALUE": None,
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

PROJECT_LS = "default" # LangSmith
ENDPOINT_LC = "https://api.smith.langchain.com" # LangChain
CLIENT_LC = smiths.Client(
  api_url=ENDPOINT_LC, api_key=os.getenv("LANGCHAIN_API_KEY")
)
TRACER_LS = smiths.LangChainTracer(project_name=PROJECT_LS, client=CLIENT_LC)
RUN_COLLECTOR = smiths.RunCollectorCallbackHandler()

#*==============================================================================

@st.cache_data(show_spinner=False)
def get_LC_run_url(run_id):
  try:
    result = CLIENT_LC.read_run(run_id).url
  except:
    result = None
    
  return result

def create_callbacks() -> list:
  st_callback = utils.StreamlitCallbackHandler(st.container())
  callbacks = [st_callback]
  return callbacks

def get_langchain_session_dynamodb_table(user: str = "admin"):
  try:
    result = asyncio.run(client.get_langchain_session_dynamodb_table(user=user))
  except:
    result = None
  return result

langchain_session_dynamodb_table = get_langchain_session_dynamodb_table(user="admin")
#*==============================================================================

async def process_on_user_input(
  prompt: str, 
):
  # Clear the container before displaying user's message
  if st.session_state.container_placeholder is not None:
    st.session_state.container_placeholder.empty()
  
  st.chat_message(CHAT_ROLE.user).markdown(prompt)
  stream = client.stream_agent_sync(prompt)
  st.chat_message(CHAT_ROLE.assistant).write_stream(stream)
  
async def render_chat_messages_on_rerun():
  chat_history = await client.get_chat_history()
  for msg in chat_history:
    msg: Union[prompts.AIMessage, prompts.HumanMessage]
    st.chat_message(msg["type"]).markdown(msg["content"])

async def on_click_btn_clear_chat_history(
  model: agents.MyAgent,
):
  await model.history.clear_chat_history()
  del st.session_state[STATES["LAST_RUN"]["KEY"]]
  st.toast(":orange[History cleared]", icon="🗑️")

def on_click_btn_new_chat(
  
):
  st.toast(":green[Chat created]", icon="✅")
  st.session_state[STATES["SELECTED_CHAT"]["KEY"]] = None
  
def on_click_btn_clear_chat(
  
):
  st.toast(":red[Chat cleared]", icon="❌")

# def on_change_box_selected_chat(
#   model: agents.MyAgent,
# ):
#   model.history.session_id = st.session_state[STATES["SELECTED_CHAT"]["KEY"]]

#*==============================================================================

containter_empty_btn_opts_holder = st.empty()

asyncio.run(render_chat_messages_on_rerun())

with st.sidebar:
  btn_new_chat = st.button(
    label="💬", 
    key=STATES["BTN_NEW_CHAT"]["KEY"],
    help="Create new Chat",
    on_click=on_click_btn_new_chat, 
    kwargs=dict()
  )
  if btn_new_chat:
    st.rerun()
  
  selected_chat = st.selectbox(
    label="Chat",
    label_visibility="collapsed",
    help="Select Your Chat",
    placeholder="Chats",
    options=langchain_session_dynamodb_table,
    key=STATES["SELECTED_CHAT"]["KEY"],
    # on_change=on_change_box_selected_chat,
  )
  
  prompt_example = st.selectbox(
    label="Examples",
    label_visibility="collapsed",
    help="Example prompts",
    placeholder="Examples",
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
  
  cols_clear = st.columns([0.25, 0.25, 0.5])
  
  btn_clear_chat_history = cols_clear[0].button(
    label="🗑️", 
    help="Clear Chat History",
    key=STATES["BTN_CLEAR_CHAT_HISTORY"]["KEY"],
  )
  if btn_clear_chat_history:
    asyncio.run(on_click_btn_clear_chat_history())
    st.rerun()
  
  btn_clear_chat = cols_clear[1].button(
    label="❌", 
    help="Clear Chat",
    key=STATES["BTN_CLEAR_CHAT"]["KEY"],
    on_click=on_click_btn_clear_chat, 
    kwargs=dict()
  )
  
#*----------------------------------------------------------------------------

prompt: Union[str, None] = st.chat_input("Say something")

if prompt_example:
  prompt = prompt_example
  del st.session_state[STATES["PROMPT_EXAMPLE"]["KEY"]]
  
if prompt:
  asyncio.run(process_on_user_input(prompt))

# st.write(st.session_state)