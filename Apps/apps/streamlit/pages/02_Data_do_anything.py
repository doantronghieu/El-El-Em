import streamlit as st
import typing_inspect
import os
import add_packages
from toolkit.langchain import document_loaders

#*==============================================================================

st.set_page_config(
  layout="wide",
  page_title="Do Anything w/ Document"
)

current_file_path = os.path.abspath(__file__)
parent_path = os.path.dirname(current_file_path)
parent_dir = parent_path.split("/")[-1]

if parent_dir != "pages":
  st.sidebar.page_link(f"main.py", label="Home")
  
  if os.getenv("STREAMLIT_GENERAL_CHAT"):
    st.sidebar.page_link(f"pages/01_general_chat.py", label="General Chat")
  if os.getenv("STREAMLIT_DATA_DO_ANYTHING"):
    st.sidebar.page_link(f"pages/02_Data_do_anything.py", label="Do Anything w/ Data")
  if os.getenv("STREAMLIT_GENERATE_ANYTHING"):
    st.sidebar.page_link(f"pages/03_Generate_anything.py", label="Generate Anything")
  if os.getenv("STREAMLIT_VTC"):
    st.sidebar.page_link(f"pages/VTC.py", label="VTC")
  
  st.sidebar.divider()

#*==============================================================================

with st.sidebar:
  st.header("Do Anything w/ Document")
  
  document_src = st.selectbox(
    label="Document Source Type",
    label_visibility="collapsed",
    placeholder="Document Source Type",
    options=list(typing_inspect.get_args(document_loaders.TypeDocumentSrc)),
    index=None,
    key="document_src",
  )
  with st.expander(label="Source"):
    st.write("Choose one")
    
    src_file = st.file_uploader(
      label="Source File",
      accept_multiple_files=False,
    )
    src_txt = st.text_input(
      label="Source Link",
      placeholder="Youtube url, Website url ..."
    )
  
  task = st.selectbox(
    label="Task",
    label_visibility="collapsed",
    placeholder="Task",
    options=[
      "QnA", "Extraction", "Classification", "Summarization",
    ],
    index=None,
    key="task",
  )