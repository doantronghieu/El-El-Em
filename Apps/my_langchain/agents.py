import uuid
import add_packages
from my_langchain import histories, runnables

from langchain.agents import (
  create_openai_tools_agent, create_openai_functions_agent, 
  create_openapi_agent, create_react_agent, create_self_ask_with_search_agent,
  AgentExecutor
)
from langchain.agents.format_scratchpad.openai_tools import (
  format_to_openai_tool_messages, 
)
from langchain.agents.format_scratchpad import (
  format_to_openai_function_messages,
)
from langchain.agents.output_parsers.openai_tools import (
  OpenAIToolsAgentOutputParser,
)
from langchain_core.agents import (
  AgentActionMessageLog, AgentFinish,
)

#*----------------------------------------------------------------------------

def invoke_agent_executor(agent_executor: AgentExecutor, input_str):

    return agent_executor.invoke({
        "input": input_str
    })["output"]

#*----------------------------------------------------------------------------


class MyAgent:
    def __init__(self, prompt, tools, agent_type, llm):
        self.prompt = prompt
        self.tools = tools
        self.agent_type = agent_type
        self.llm = llm
        self.session_id = str(uuid.uuid4())  # Generate a UUID for session_id

        self.agent = self._create_agent()
        self.agent_executor = AgentExecutor(
            agent=self.agent, tools=self.tools, verbose=True,
            handle_parsing_errors=True,
        )

        self.message_history = histories.ChatMessageHistory()
        self.agent_with_chat_history = runnables.RunnableWithMessageHistory(
            self.agent_executor,
            lambda session_id: self.message_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

    def _create_agent(self):
        if self.agent_type == "openai_tools":
            return create_openai_tools_agent(self.llm, self.tools, self.prompt)
        elif self.agent_type == "react":
            return create_react_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
        else:
            raise ValueError(
                "Invalid agent type. Supported types are 'openai_tools' and 'react'.")

    def invoke_agent(self, input_message):
        config = {"configurable": {"session_id": self.session_id}}
        return self.agent_with_chat_history.invoke({"input": input_message}, config=config)['output']


