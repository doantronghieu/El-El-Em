from typing import Union, Sequence
import asyncio
import uuid
import os
import time

from regex import P
from toolkit.langchain import models

import add_packages

import streamlit as st
from streamlit_feedback import streamlit_feedback

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.tools import BaseTool
from langchain_core.runnables import Runnable

from toolkit.langchain import (
  agent_tools, prompts, agents, smiths, runnables, memories
)

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

def hash_func_ChatHistory(obj: agents.ChatHistory):
  return obj.session_id

#*==============================================================================

@st.cache_data(show_spinner=False)
def get_LC_run_url(run_id):
  try:
    result = CLIENT_LC.read_run(run_id).url
  except:
    result = None
    
  return result

@st.cache_resource(hash_funcs={agents.ChatHistory: hash_func_ChatHistory})
def create_agent(
  _llm: BaseLanguageModel,
  _tools: Sequence[BaseTool],
  _prompt: ChatPromptTemplate,
  history: agents.ChatHistory,
  _agent_type: str,
) -> Runnable:
  agent = agents.MyAgent(
    llm=_llm,
    tools=_tools,
    prompt=_prompt,
    history=history,
    agent_type=_agent_type,
  )
  return agent

@st.cache_resource
def get_langchain_session_dynamodb_table():
  return memories.LangChainSessionDynamodbTable()

def create_callbacks() -> list:
  st_callback = utils.StreamlitCallbackHandler(st.container())
  callbacks = [st_callback, TRACER_LS, RUN_COLLECTOR]
  return callbacks

#*---------------------------------------------------------------------------

langchain_session_dynamodb_table = get_langchain_session_dynamodb_table()

llm = models.chat_openai
tools = [
  agent_tools.TavilySearchResults(max_results=3)
]
prompt = prompts.create_prompt_tool_calling_agent()

history = agents.ChatHistory(
  history_type='dynamodb',
  user_id="admin",
  session_id=st.session_state[STATES["SELECTED_CHAT"]["KEY"]] \
    if st.session_state[STATES["SELECTED_CHAT"]["KEY"]] else None,
)

agent: agents.MyAgent = create_agent(
  _llm=llm,
  _tools=tools,
  _prompt=prompt,
  history=history,
  _agent_type="tool_calling",
)

#*==============================================================================


async def generate_response(
  input: str, 
  model: agents.MyAgent,
):
  response = await model.invoke_agent(
    input_message=input, 
    callbacks=create_callbacks(),
    mode="sync",
  )
  
  st.session_state[STATES["LAST_RUN"]["KEY"]] = RUN_COLLECTOR.traced_runs[0].id
  
  return response

def render_last_msg_opt_btns():
  if agent.history.chat_history:
    # Create a new container and store its placeholder
    st.session_state.container_placeholder = st.empty()
    
    with st.session_state.container_placeholder:
      cols_last_msg_opts = st.columns([0.94, 0.03, 0.03])
      cols_last_msg_opts[1].button("↻", key="btn_lst_msg_regenerate")
      cols_last_msg_opts[2].button("📋", key="btn_lst_msg_copy")

async def process_on_user_input(
  prompt: str, 
  model: agents.MyAgent,
):
  # Clear the container before displaying user's message
  if st.session_state.container_placeholder is not None:
      st.session_state.container_placeholder.empty()
  
  st.chat_message(CHAT_ROLE.user).markdown(prompt)
  stream = await generate_response(prompt, model)
  st.chat_message(CHAT_ROLE.assistant).write(stream)
  
  # render_last_msg_opt_btns()
  
async def render_chat_messages_on_rerun(
  model: agents.MyAgent,
):
  for msg in await model.history._get_chat_history():
    msg: Union[prompts.AIMessage, prompts.HumanMessage]
    st.chat_message(msg.type).markdown(msg.content)

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

def on_change_box_selected_chat(
  model: agents.MyAgent,
):
  model.history.session_id = st.session_state[STATES["SELECTED_CHAT"]["KEY"]]

#*==============================================================================

containter_empty_btn_opts_holder = st.empty()

asyncio.run(render_chat_messages_on_rerun(agent))

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
    options=langchain_session_dynamodb_table.get_session_ids("admin"),
    key=STATES["SELECTED_CHAT"]["KEY"],
    on_change=on_change_box_selected_chat,
    kwargs=dict(model=agent),
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
    asyncio.run(on_click_btn_clear_chat_history(agent))
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
  asyncio.run(process_on_user_input(prompt, agent))

#*------------------------------------------------------------------------------
# Feedback

if st.session_state[STATES["LAST_RUN"]["KEY"]]:
  run_url = get_LC_run_url(st.session_state[STATES["LAST_RUN"]["KEY"]])
  
  if run_url is None:
    pass
  
  feedback = streamlit_feedback(
    feedback_type="faces",
    optional_text_label="[Optional] Please provide an explanation",
    key=f'feedback_{st.session_state[STATES["LAST_RUN"]["KEY"]]}'
  )
  
  if feedback:
    scores = {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0}
    
    CLIENT_LC.create_feedback(
      st.session_state[STATES["LAST_RUN"]["KEY"]],
      feedback["type"],
      score=scores[feedback["score"]],
      comment=feedback.get("text", None)
    )
    
    st.toast("Feedback recorded.", icon="📝")

st.write(st.session_state)