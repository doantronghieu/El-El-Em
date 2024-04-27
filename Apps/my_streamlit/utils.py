import streamlit as st
import typing
import time
from pydantic import BaseModel, Field
from typing import Any
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

#*==============================================================================

STATES = {
  "MY_KEY": {
    "INITIAL_VALUE": 0,
  },
}

def initialize_session_state(STATES: dict[str, dict[str, typing.Any]]):
  """
  Function to initialize session state with default values
  """
  for state_name, state_info in STATES.items():
    key = state_name.lower()  # Set key as the state name
    if "KEY" not in state_info:
      state_info["KEY"] = key  # If KEY not provided, use state name as key
    if state_info["INITIAL_VALUE"] != 'widget' and state_info["KEY"] not in st.session_state:
      st.session_state[state_info["KEY"]] = state_info["INITIAL_VALUE"]


def retain_session_state(STATES: dict[str, dict[str, typing.Any]]):
  """
  Function to retain session state for specific keys.\n
  Add to the end of the script.
  """
  for state_name, state_info in STATES.items():
    st.session_state[state_info["KEY"]] = st.session_state[state_info["KEY"]]


def store_value(key: str):
  st.session_state[key] = st.session_state[f"_{key}"]
  
def load_value(key: str):
  st.session_state[f"_{key}"] = st.session_state[key]

#*==============================================================================

class ChatRole:
  def __init__(self) -> None:
    self.user = "user"
    self.assistant = "assistant"

CHAT_ROLE = ChatRole()

class MsgItem:
  def __init__(self) -> None:
    self.role = "role"
    self.content = "content"
    
MSG_ITEM = MsgItem()

#*==============================================================================

