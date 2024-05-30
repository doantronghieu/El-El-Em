import streamlit as st
import os
import add_packages
#*==============================================================================

st.set_page_config(
  layout="wide",
  page_title="Home"
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

st.markdown(
"""
# Welcome!

Hey there! Welcome to our chatbot app. I'm here to assist you with any questions you have.

Feel free to ask me anything, whether it's about our products, services, or anything else you're curious about. I'm here to help!

If you need assistance, just type your question, and I'll do my best to provide you with a helpful response.

Let's get started!
"""
)

