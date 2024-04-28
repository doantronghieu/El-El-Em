import uuid
import add_packages
from loguru import logger
from typing import Union, Optional, List, Literal

from my_langchain import histories, runnables

from langchain.agents import (
  create_openai_tools_agent, create_openai_functions_agent, 
  create_react_agent, create_self_ask_with_search_agent,
  create_xml_agent, create_tool_calling_agent,
  AgentExecutor
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts.chat import BaseChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.agents import (
  AgentActionMessageLog, AgentFinish, AgentAction
)
from langchain_core.messages import AIMessage, HumanMessage, ChatMessage

from langchain.agents.openai_assistant import OpenAIAssistantRunnable
from langchain.agents.format_scratchpad.openai_tools import (
  format_to_openai_tool_messages, 
)
from langchain.agents.format_scratchpad import (
  format_to_openai_function_messages,
)
from langchain.agents.output_parsers.openai_tools import (
  OpenAIToolsAgentOutputParser,
)

#*----------------------------------------------------------------------------

def invoke_agent_executor(agent_executor: AgentExecutor, input_str):

    return agent_executor.invoke({
        "input": input_str
    })["output"]

#*----------------------------------------------------------------------------

AgentType = Literal["tool_calling", "openai_tools", "react", "anthropic"]

class MyAgent:
  def __init__(
    self, 
    llm: Union[BaseChatModel, None],
    tools: list[BaseTool],
    prompt: Union[BaseChatPromptTemplate, None],
    agent_type: AgentType, 
  ):
    self.prompt = prompt
    self.tools = tools
    self.agent_type = agent_type
    self.llm = llm
    self.session_id = str(uuid.uuid4())  # Generate a UUID for session_id
    self.config = {"configurable": {"session_id": self.session_id}}

    self.agent = self._create_agent()
    self.agent_executor = AgentExecutor(
      agent=self.agent, tools=self.tools, verbose=True,
      handle_parsing_errors=True,
      return_intermediate_steps=False,
    )
    self.chat_history: list[Union[AIMessage, HumanMessage, ChatMessage, None]] = []

    self.message_history = histories.ChatMessageHistory()
    self.agent_executor_conversable = runnables.RunnableWithMessageHistory(
      self.agent_executor,
      lambda session_id: self.message_history,
      input_messages_key="input",
      output_messages_key="output",
      history_messages_key="chat_history",
    )

  def _create_agent(self) -> Runnable:
    logger.info(f"Agent type: {self.agent_type}")
    
    if self.agent_type == "tool_calling":
      return create_tool_calling_agent(self.llm, self.tools, self.prompt)
    elif self.agent_type == "openai_tools":
      return create_openai_tools_agent(self.llm, self.tools, self.prompt)
    elif self.agent_type == "react":
      return create_react_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
    elif self.agent_type == "anthropic": # todo
      return create_xml_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)
    else:
      raise ValueError(
        "Invalid agent type. Supported types are 'openai_tools' and 'react'.")

  def _add_messages_to_history(
    self, 
    msg_user: str, 
    msg_ai: str,
  ):
    
    self.chat_history.append(HumanMessage(msg_user))
    self.chat_history.append(AIMessage(msg_ai))

  def clear_chat_history(self):
    self.chat_history = []
  
  def invoke_agent(
    self, 
    input_message: str, 
    callbacks: Optional[List] = None,
  ):
    input_data = {"input": input_message, "chat_history": self.chat_history}
    configs = {"callbacks": callbacks} if callbacks else {}
    
    result = self.agent_executor.invoke(input_data, configs)
    result = result["output"]
    
    self._add_messages_to_history(input_message, result)
    
    return result
  
  async def stream_agent(
    self, 
    input_message: str, 
    callbacks: Optional[List] = None,
  ):
    """
    Usage: await agent.stream_agent(input_message)
    """
    result = ''
    
    async for event in self.agent_executor.astream_events(
      {
        "input": input_message,
        "chat_history": self.chat_history,
      },
      version="v1",
    ):
      kind = event["event"]
      if kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
          # Empty content in the context of OpenAI means
          # that the model is asking for a tool to be invoked.
          # So we only print non-empty content
          print(content, end="")
          result += content
          # yield content
    
    print()
    
    self._add_messages_to_history(input_message, result)
    
    return result
  
  def invoke_conversable_agent(self, input_message: str):
    return self.agent_executor_conversable.invoke({"input": input_message}, config=self.config)['output']
  
  async def stream_conversable_agent(self, input_message: str):
    """
    Usage: await agent.stream_conversable_agent(input_message)
    """
    async for event in self.agent_executor_conversable.astream_events(
      {"input": input_message}, config=self.config, version="v1"
    ):
      kind = event["event"]
      if kind == "on_chat_model_stream":
        content = event["data"]["chunk"].content
        if content:
          # Empty content in the context of OpenAI means
          # that the model is asking for a tool to be invoked.
          # So we only print non-empty content
          print(content, end="")
          # yield content
    
    print()

"""
async def bot_APP(chat_history, human_msg):
  chat_history[-1][1] = ""

  async for event in AGENT_INSTANCE.agent.agent_executor_conversable.astream_events(
    {"input": human_msg}, config=AGENT_INSTANCE.agent.config, version="v1"
  ):
    kind = event["event"]
    if kind == "on_chat_model_stream":
      content = event["data"]["chunk"].content
      if content:
        chat_history[-1][1] += content
        yield chat_history
"""
