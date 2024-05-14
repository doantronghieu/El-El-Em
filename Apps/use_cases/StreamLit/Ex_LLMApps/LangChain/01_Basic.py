import streamlit as st
import add_packages

from toolkit.langchain import chat_models

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

def generate_response(input):
  response = chat_models.chat_openai.stream(input)

  return response

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
  
  stream = generate_response(prompt)

  response = st.chat_message(CHAT_ROLE.assistant).write_stream(stream)
  
  st.session_state[STATES["MESSAGES"]["KEY"]].append({
    MSG_ITEM.role: CHAT_ROLE.user, MSG_ITEM.content: prompt
  })
  st.session_state[STATES["MESSAGES"]["KEY"]].append({
    MSG_ITEM.role: CHAT_ROLE.assistant, MSG_ITEM.content: response
  })

# st.write(st.session_state)