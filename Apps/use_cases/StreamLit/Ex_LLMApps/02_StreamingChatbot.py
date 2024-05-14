import streamlit as st
import add_packages
import time
import random

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

def generate_response():
  response = random.choice(
    [
      "Hello there! How can I assist you today?",
      "Hi, human! Is there anything I can help you with?",
      "Do you need help?",
    ]
  )
  for word in response.split():
    yield f"{word} "
    time.sleep(0.05)

#*==============================================================================

# Store chat history in list(dict), append messages for user or bot.
# List entries: role (author) and content (message).

# Display chat messages from history on app rerun
for msg in st.session_state[STATES["MESSAGES"]["KEY"]]:
  with st.chat_message(msg[MSG_ITEM.role]):
    st.markdown(msg[MSG_ITEM.content])

prompt = st.chat_input("Say something")

if prompt:
  st.chat_message(CHAT_ROLE.user).markdown(prompt)
  
  stream = generate_response()

  response = st.chat_message(CHAT_ROLE.assistant).write_stream(stream)
  
  st.session_state[STATES["MESSAGES"]["KEY"]].append({
    MSG_ITEM.role: CHAT_ROLE.user, MSG_ITEM.content: prompt
  })
  st.session_state[STATES["MESSAGES"]["KEY"]].append({
    MSG_ITEM.role: CHAT_ROLE.assistant, MSG_ITEM.content: response
  })

# st.write(st.session_state)