import uuid
from typing import AsyncGenerator
import add_packages
from loguru import logger
from typing import Union, Optional, List, Literal

from toolkit.langchain import histories, runnables

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

class MyAgent:
  def __init__(
    self, 
    llm: Union[BaseChatModel, None],
    tools: list[BaseTool],
    prompt: Union[BaseChatPromptTemplate, None],
    agent_type: Literal[
      "tool_calling", "openai_tools", "react", "anthropic"
    ] = "tool_calling", 
    agent_verbose: bool = False,
    history_type: Literal["in_memory", "dynamodb"] = "in_memory",
  ):
    self.llm = llm
    self.tools = tools
    self.prompt = prompt
    
    self.agent_type = agent_type
    self.agent_verbose = agent_verbose
    
    self.history_type = history_type
    
    self.session_id = str(uuid.uuid4())  # Generate a UUID for session_id
    self.config = {"configurable": {"session_id": self.session_id}}

    self.agent = self._create_agent()
    self.agent_executor = AgentExecutor(
      agent=self.agent, tools=self.tools, verbose=self.agent_verbose,
      handle_parsing_errors=True,
      return_intermediate_steps=False,
    )
    self.chat_history: list[Union[AIMessage, HumanMessage, ChatMessage, None]] = []

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
  
  async def invoke_agent(
    self, 
    input_message: str, 
    callbacks: Optional[List] = None,
    mode: Literal["sync", "async"] = "sync",
  ):
    result = None
  
    input_data = {"input": input_message, "chat_history": self.chat_history}
    
    configs = {}
    configs["callbacks"] = callbacks if callbacks else []
    
    if mode == "sync":
      result = self.agent_executor.invoke(input_data, configs)
    elif mode == "async":
      result = await self.agent_executor.ainvoke(input_data, configs)

    result = result["output"]
    
    self._add_messages_to_history(input_message, result)

    return result

  async def astream_events_basic(
    self,
    input_message: str,
    show_tool_call: bool = False,
  ) -> AsyncGenerator[str, None]:
    """
    async for chunk in agent.astream_events_basic("Hello"):
      print(chunk, end="", flush=True)
    """
  
    result = ""
    async for event in self.agent_executor.astream_events(
      input={"input": input_message, "chat_history": self.chat_history},
      version="v1",
    ):
      event_event = event["event"]
      event_name = event["name"]

      if event["event"] == "on_chat_model_stream":
        chunk = dict(event["data"]["chunk"])["content"]
        result += chunk
        yield chunk

      if show_tool_call and event_event == "on_chain_stream" and event_name == "Agent":
        if 'actions' in event['data']['chunk']:
          event_log = dict(list(event['data']['chunk']['actions'])[0])['log']
          chunk = event_log
          result += chunk
          yield chunk

    self._add_messages_to_history(input_message, result)

  async def astream_events_basic_wrapper(
    self,
    input_message: str,
  ):
    result = ""
    async for chunk in self.astream_events_basic(input_message):
        result += chunk
        print(chunk, end="", flush=True)
    return result

  def hello():
    ...