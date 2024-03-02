from langchain_experimental.plan_and_execute import (load_chat_planner,
                                                     load_agent_executor, PlanAndExecute)
from langchain.chat_models import ChatOpenAI
from langchain.chains.base import Chain
from langchain.agents import AgentExecutor, initialize_agent, load_tools, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from typing import Literal
import streamlit as st
from config import set_environment
set_environment()


# LANGCHAIN ####################################################################

ReasoningStrategies = Literal['zero-shot-react', 'plan-and-solve']


def load_agent(tool_names: list[str],
               strategy: ReasoningStrategies = 'zero-shot-react') -> Chain:
    llm = ChatOpenAI(temperature=0, streaming=True)
    tools = load_tools(tool_names=tool_names, llm=llm)

    if strategy == 'plan-and-solve':
        planner = load_chat_planner(llm)
        executor = load_agent_executor(llm, tools, verbose=True)
        return PlanAndExecute(planner=planner, executor=executor, verbose=True)

    return initialize_agent(tools=tools, llm=llm,
                            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                            verbose=True)
# STREAMLIT ####################################################################

strategy = st.radio(label='Reasoning Strategy',
                    options=('plan-and-solve', 'zero-shot-react'))

tool_names = st.multiselect('Which tools do you want to use?',
                            [
                                "google-search", "ddg-search", "wolfram-alpha", "arxiv",
                                "wikipedia", "python_repl", "pal-math", "llm-math"
                            ],
                            [
                                "ddg-search", "wolfram-alpha", "wikipedia"
                            ])

agent_chain = load_agent(tool_names=tool_names, strategy=strategy)

st_callback = StreamlitCallbackHandler(st.container())

if prompt := st.chat_input():
    st.chat_message('user').write(prompt)

    with st.chat_message('assistant'):
        st_callback = StreamlitCallbackHandler(st.container())
        response = agent_chain.run(prompt, callbacks=[st_callback])
        st.write(response)

# conda activate chatbot_env
# streamlit run Building_Capable_Assistants.py
