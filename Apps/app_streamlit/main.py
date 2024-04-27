import streamlit as st

#*==============================================================================

st.set_page_config(
  layout="wide",
)

st.sidebar.page_link(f"main.py", label="Home")
st.sidebar.page_link(f"pages/1_general_chat.py", label="General Chat")

#*==============================================================================

st.markdown(
"""
# Welcome to My Chatbot App

Hey there! Welcome to our chatbot app. I'm here to assist you with any questions you have.

Feel free to ask me anything, whether it's about our products, services, or anything else you're curious about. I'm here to help!

If you need assistance, just type your question, and I'll do my best to provide you with a helpful response.

Let's get started!
"""
)

