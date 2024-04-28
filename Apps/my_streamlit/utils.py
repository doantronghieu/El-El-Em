import streamlit as st
from datetime import datetime
import typing
import time
import uuid
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

def generate_user_uuid(user_email: str):
	name = f"{user_email}"
	return str(uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name=name))

def generate_chat_uuid(app_name: str, user_id: str):
	name = f"{app_name}_{user_id}_{datetime.now()}"
	return str(uuid.uuid5(namespace=uuid.NAMESPACE_DNS, name=name))