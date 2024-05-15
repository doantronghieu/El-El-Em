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

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypeVar, Union
from uuid import UUID

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult



class TokenByTokenHandler(AsyncCallbackHandler):
	def __init__(self, tags_of_interest: List[str]) -> None:
		"""
		Custom handler that will print the tokens to stdout.
		Instead of printing to stdout you can send the data elsewhere; e.g., to a 
		streaming API response
  
		Args:
				tags_of_interest: Only LLM tokens from models with these tags will be
													printed.
		"""
		self.tags_of_interest = tags_of_interest

	async def on_chain_start(
		self,
		serialized: Dict[str, Any],
		inputs: Dict[str, Any],
		*,
		run_id: UUID,
		parent_run_id: Optional[UUID] = None,
		tags: Optional[List[str]] = None,
		metadata: Optional[Dict[str, Any]] = None,
		**kwargs: Any,
	) -> None:
		"""Run when chain starts running."""
		# print("on chain start: ")
		# print(inputs)

	async def on_chain_end(
		self,
		outputs: Dict[str, Any],
		*,
		run_id: UUID,
		parent_run_id: Optional[UUID] = None,
		tags: Optional[List[str]] = None,
		**kwargs: Any,
	) -> None:
		"""Run when chain ends running."""
		# print("On chain end")
		# print(outputs)

	async def on_chat_model_start(
		self,
		serialized: Dict[str, Any],
		messages: List[List[BaseMessage]],
		*,
		run_id: UUID,
		parent_run_id: Optional[UUID] = None,
		tags: Optional[List[str]] = None,
		metadata: Optional[Dict[str, Any]] = None,
		**kwargs: Any,
	) -> Any:
		"""Run when a chat model starts running."""
		overlap_tags = self.get_overlap_tags(tags)

		# if overlap_tags:
		# 	print(",".join(overlap_tags), end=": ", flush=True)

	def on_tool_start(
		self,
		serialized: Dict[str, Any],
		input_str: str,
		*,
		run_id: UUID,
		parent_run_id: Optional[UUID] = None,
		tags: Optional[List[str]] = None,
		metadata: Optional[Dict[str, Any]] = None,
		inputs: Optional[Dict[str, Any]] = None,
		**kwargs: Any,
	) -> Any:
		"""Run when tool starts running."""
		print(f"Tool: {serialized}")

	def on_tool_end(
		self,
		output: Any,
		*,
		run_id: UUID,
		parent_run_id: Optional[UUID] = None,
		**kwargs: Any,
	) -> Any:
		"""Run when tool ends running."""
		print(f"Result: {str(output)}")

	async def on_llm_end(
		self,
		response: LLMResult,
		*,
		run_id: UUID,
		parent_run_id: Optional[UUID] = None,
		tags: Optional[List[str]] = None,
		**kwargs: Any,
	) -> None:
		"""Run when LLM ends running."""
		overlap_tags = self.get_overlap_tags(tags)

		if overlap_tags:
			# Who can argue with beauty?
			print()
			# print()

	def get_overlap_tags(self, tags: Optional[List[str]]) -> List[str]:
		"""Check for overlap with filtered tags."""
		if not tags:
			return []
		return sorted(set(tags or []) & set(self.tags_of_interest or []))

	async def on_llm_new_token(
		self,
		token: str,
		*,
		chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
		run_id: UUID,
		parent_run_id: Optional[UUID] = None,
		tags: Optional[List[str]] = None,
		**kwargs: Any,
) -> None:
		"""Run on new LLM token. Only available when streaming is enabled."""
		overlap_tags = self.get_overlap_tags(tags)

		if token and overlap_tags:
			print(token, end="", flush=True)

class MyAgent:
	def __init__(
		self, 
		llm: Union[BaseChatModel, None],
		tools: list[BaseTool],
		prompt: Union[BaseChatPromptTemplate, None],
		agent_type: Literal["tool_calling", "openai_tools", "react", "anthropic"] = "tool_calling", 
		agent_verbose: bool = False,
	):
		self.prompt = prompt
		self.tools = tools
		self.agent_type = agent_type
		self.agent_verbose = agent_verbose
  
		self.llm = llm.with_config({"tags": ["agent_llm"]})
		self.session_id = str(uuid.uuid4())  # Generate a UUID for session_id
		self.config = {"configurable": {"session_id": self.session_id}}

		self.token_handler = TokenByTokenHandler(
			tags_of_interest=["tool_llm", "agent_llm"]
    )
  
		self.agent = self._create_agent()
		self.agent_executor = AgentExecutor(
			agent=self.agent, tools=self.tools, verbose=self.agent_verbose,
			handle_parsing_errors=True,
			return_intermediate_steps=False,
		).with_config(
			{"run_name": "Agent"}
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
		# configs["callbacks"].append(self.token_handler)
		
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