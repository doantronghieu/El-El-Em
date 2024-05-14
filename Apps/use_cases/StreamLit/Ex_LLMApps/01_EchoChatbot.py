import streamlit as st
import add_packages
from typing import List

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
}

utils.initialize_session_state(STATES)
#*==============================================================================


# chat_user = st.chat_message(CHAT_ROLE.user)
# chat_assistant = st.chat_message(CHAT_ROLE.assistant)
# chat_input = st.chat_input("Say something")

# chat_user.write("Hello")
# chat_assistant.write("Hello Human")

# if chat_input:
#   st.write(f"User has sent the following prompt: {chat_input}")
#*==============================================================================

# Store chat history in list(dict), append messages for user or bot.
# List entries: role (author) and content (message).

for msg in st.session_state[STATES["MESSAGES"]["KEY"]]:
  with st.chat_message(msg[MSG_ITEM.role]):
    st.markdown(msg[MSG_ITEM.content])

prompt = st.chat_input("Say something")

if prompt:
  st.chat_message(CHAT_ROLE.user).markdown(prompt)
  
  response = f"Echo: {prompt}"
  st.chat_message(CHAT_ROLE.assistant).markdown(response)
  
  st.session_state[STATES["MESSAGES"]["KEY"]].append({
    MSG_ITEM.role: CHAT_ROLE.user, MSG_ITEM.content: prompt
  })
  st.session_state[STATES["MESSAGES"]["KEY"]].append({
    MSG_ITEM.role: CHAT_ROLE.assistant, MSG_ITEM.content: response
  })