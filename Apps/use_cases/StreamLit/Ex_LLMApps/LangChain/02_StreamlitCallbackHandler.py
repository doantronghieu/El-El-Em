import streamlit as st
import add_packages

from toolkit.langchain import chat_models, agent_tools, prompts, agents

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

st_callback = utils.StreamlitCallbackHandler(st.container())

#*==============================================================================

llm = chat_models.chat_openai

tool_search = agent_tools.TavilySearchResults(max_results=3)
tools = [
  tool_search,
]

prompt = prompts.hub.pull("hwchase17/react")

agent = agents.create_react_agent(llm=llm, tools=tools, prompt=prompt)
agent_executor = agents.AgentExecutor(agent=agent, tools=tools, verbose=True)

#*==============================================================================

def generate_response(input):
  response = agent_executor.invoke(
    {"input": input},
    {"callbacks": [st_callback]}
  )

  return response["output"]

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

  response = st.chat_message(CHAT_ROLE.assistant).write(stream)
  
  st.session_state[STATES["MESSAGES"]["KEY"]].append({
    MSG_ITEM.role: CHAT_ROLE.user, MSG_ITEM.content: prompt
  })
  st.session_state[STATES["MESSAGES"]["KEY"]].append({
    MSG_ITEM.role: CHAT_ROLE.assistant, MSG_ITEM.content: response
  })

# st.write(st.session_state)