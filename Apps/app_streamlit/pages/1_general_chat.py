import streamlit as st
import add_packages

#*==============================================================================

st.set_page_config(
  layout="wide",
)

st.sidebar.page_link(f"main.py", label="Home")
st.sidebar.page_link(f"pages/1_general_chat.py", label="General Chat")

#*==============================================================================

st.markdown("# Sub Page 🎉")
st.sidebar.markdown("# Sub Page 🎉")