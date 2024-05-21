import streamlit as st
import typing_inspect
import os
import add_packages
from toolkit.langchain import document_loaders

#*==============================================================================

st.set_page_config(
  layout="wide",
  page_title="Generate Anything"
)

current_file_path = os.path.abspath(__file__)
parent_path = os.path.dirname(current_file_path)
parent_dir = parent_path.split("/")[-1]

if parent_dir != "pages":
  st.sidebar.page_link(f"main.py", label="Home")
  st.sidebar.page_link(f"pages/01_general_chat.py", label="General Chat")
  st.sidebar.page_link(f"pages/02_Doc_do_anything.py", label="Do Anything w/ Document")
  st.sidebar.page_link(f"pages/03_Generate_anything.py", label="Generate Anything")
  st.sidebar.divider()

#*==============================================================================

with st.sidebar:
  st.header("Generate Anything")
  
  task = st.selectbox(
    label="Task",
    label_visibility="collapsed",
    placeholder="Task",
    options=[
      "Newsletter",
    ],
    index=None,
    key="task",
  )