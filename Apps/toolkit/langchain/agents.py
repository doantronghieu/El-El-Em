import add_packages
import boto3
from loguru import logger
from typing import Union, Optional, List, Literal, AsyncGenerator, TypeAlias
from pydantic import BaseModel

from toolkit import utils

from langchain.agents import (
	create_openai_tools_agent, create_openai_functions_agent, 
	create_react_agent, create_self_ask_with_search_agent,
	create_xml_agent, create_tool_calling_agent,
	AgentExecutor
)
from langchain_community.agent_toolkits import create_sql_agent

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

from langchain_community.chat_message_histories.dynamodb import DynamoDBChatMessageHistory

#*==============================================================================

dynamodb = boto3.resource("dynamodb")

#*==============================================================================

TypeHistoryType: TypeAlias = Literal["in_memory", "dynamodb"]
TypeUserId: TypeAlias = str
TypeSessionId: TypeAlias = Union[str, None]

def create_md_txt_color(input: str, color: str="red"):
	result = f'<span style="color:{color};">{input}</span>'
	return result

class SchemaChatHistory(BaseModel):
	history_type: TypeHistoryType = "in_memory"
	user_id: TypeUserId = "admin"
	session_id: TypeSessionId = None
	history_size: Union[int, None] = 20

class ChatHistory:
	def __init__(self, schema: SchemaChatHistory):
		self.history_type = schema.history_type

		self.user_id = schema.user_id
		self.is_new_session = not bool(schema.session_id)
		self.session_id = schema.session_id if schema.session_id else utils.generate_unique_id("uuid_name")

		self.history_size = schema.history_size
	
		if self.history_type == "in_memory":
			self.chat_history = []
		elif self.history_type == "dynamodb":
			self.chat_history = DynamoDBChatMessageHistory(
				table_name="LangChainSessionTable", 
				session_id=self.session_id,
				key={
					"SessionId": self.session_id,
					"UserId": self.user_id,
				},
				history_size=self.history_size,
			)

		if self.is_new_session:
			welcome_msg = "Hello! How can I help you today?"
	
			if self.history_type == "in_memory":
				self.chat_history.append(AIMessage(welcome_msg))
			elif self.history_type == "dynamodb":
				self.chat_history.add_ai_message(welcome_msg)

		logger.info(f"User Id: {self.user_id}")
		logger.info(f"Session Id: {self.session_id}")
		logger.info(f"History Type: {self.history_type}")
	
	async def _add_messages_to_history(
		self,
		msg_user: str,
		msg_ai: str,
	):
		if self.history_type == "in_memory":
			if msg_user:
				self.chat_history.append(HumanMessage(msg_user))
			if msg_ai:
				self.chat_history.append(AIMessage(msg_ai))
		elif self.history_type == "dynamodb":
			if msg_user:
				await self.chat_history.aadd_messages(messages=[HumanMessage(msg_user)])
			if msg_ai:
				await self.chat_history.aadd_messages(messages=[AIMessage(msg_ai)])

	async def _get_chat_history(self):
		if self.history_type == "in_memory":
			return self.chat_history
		elif self.history_type == "dynamodb":
			return self.chat_history.messages

	async def clear_chat_history(self):
		if self.history_type == "in_memory":
			self.chat_history = []
		elif self.history_type == "dynamodb":
			await self.chat_history.aclear()

	async def _truncate_chat_history(
		self,
	):
		if self.history_type == "in_memory":
			self.chat_history = self.chat_history[-self.history_size:]
		elif self.history_type == "dynamodb":
			...

class MyStatelessAgent:
	def __init__(
		self,
		llm: Union[BaseChatModel, None],
		tools: list[BaseTool],
		prompt: Union[BaseChatPromptTemplate, None],
	
		agent_type: Literal[
			"tool_calling", "openai_tools", "react", "anthropic"
		] = "tool_calling",
		agent_verbose: bool = False,
	):
		self.llm = llm
		self.my_tools = tools
		self.prompt = prompt

		self.agent_type = agent_type
		self.agent_verbose = agent_verbose
	
		self.agent = self._create_agent()
		self.agent_executor = AgentExecutor(
			agent=self.agent, tools=self.my_tools, verbose=self.agent_verbose,
			handle_parsing_errors=True,
			return_intermediate_steps=False,
		)

	def _create_agent(self) -> Runnable:
		logger.info(f"Agent type: {self.agent_type}")
	
		if self.agent_type == "tool_calling":
			return create_tool_calling_agent(self.llm, self.my_tools, self.prompt)
		elif self.agent_type == "openai_tools":
			return create_openai_tools_agent(self.llm, self.my_tools, self.prompt)
		elif self.agent_type == "react":
			return create_react_agent(llm=self.llm, tools=self.my_tools, prompt=self.prompt)
		elif self.agent_type == "anthropic": # todo
			return create_xml_agent(llm=self.llm, tools=self.my_tools, prompt=self.prompt)
		else:
			raise ValueError(
					"Invalid agent type. Supported types are 'openai_tools' and 'react'.")

	def _create_chat_history(
		self,
		history_type: TypeHistoryType = "dynamodb",
		user_id: TypeUserId = "admin",
		session_id: TypeSessionId = None,
		history_size: Union[int, None] = 20,
	) -> ChatHistory:
	
		return ChatHistory(schema=SchemaChatHistory(
			history_type=history_type, user_id=user_id, session_id=session_id,
			history_size=history_size,
		)) 
	
	async def _add_messages_to_history(
		self,
		history: ChatHistory,
		history_type: TypeHistoryType,
		msg_user: str,
		msg_ai: str,
	):
		await history._add_messages_to_history(msg_user, msg_ai)
		if history_type == "in_memory":
			await history._truncate_chat_history()
	
	async def invoke_agent(
		self,
		input_message: str,
		callbacks: Optional[List] = None,
		mode: Literal["sync", "async"] = "async",
	
		history_type: TypeHistoryType = "dynamodb",
		user_id: TypeUserId = "admin",
		session_id: TypeSessionId = None,
	
		history_size: Union[int, None] = 20,
	):
		result = None

		history = self._create_chat_history(
			history_type, user_id, session_id, history_size,
		)

		input_data = {
			"input": input_message, "chat_history": await history._get_chat_history()
		}

		configs = {}
		configs["callbacks"] = callbacks if callbacks else []

		if mode == "sync":
			result = self.agent_executor.invoke(input_data, configs)
		elif mode == "async":
			result = await self.agent_executor.ainvoke(input_data, configs)

		result = result["output"]

		await self._add_messages_to_history(
			history=history,
			history_type=history_type,
			msg_user=input_message,
			msg_ai=result,
		)
	
		return result

	async def astream_events_basic(
		self,
		input_message: str,

		history_type: TypeHistoryType = "dynamodb",
		user_id: TypeUserId = "admin",
		session_id: TypeSessionId = None,
	
		show_tool_call: bool = False,
		history_size: Union[int, None] = 20,
	) -> AsyncGenerator[str, None]:
		"""
		async for chunk in agent.astream_events_basic("Hello"):
			print(chunk, end="", flush=True)
		"""

		history = self._create_chat_history(
			history_type, user_id, session_id, history_size,
		)

		result = ""

		""" used for debugging
		a = agent.events
		a = [x for x in agent.events if x["event"] == "on_chat_model_stream"]
		a_data = [x["data"] for x in a]
		a_data_chunk = [x["chunk"] for x in a_data]
		a_data_chunk_tool = [x for x in a if dict(a_data_chunk)["tool_call_chunks"]]
		a_metadata_sql_chain = [x for x in a if "..." in x["metadata"].keys()]
		"""
  
		# self.events = [] # debug
  
		async for event in self.agent_executor.astream_events(
			input={"input": input_message, "chat_history": await history._get_chat_history()},
			version="v2",
		):
			# self.events.append(event) # debug
			event_event = event["event"]
			event_name = event["name"]
	
			try: event_data_chunk = event["data"]["chunk"]
			except: pass
			
			if event_event == "on_chat_model_stream":
				chunk = dict(event_data_chunk)["content"]
		
				if (event.get("metadata", {}).get("ls_stop") == ['\nSQLResult:']) \
						or ("is_my_sql_chain_run" in event.get("metadata", {})):
					continue

				result += chunk
				res += chunk
				yield chunk
		
			if show_tool_call and event_event == "on_chain_stream":
				if event_name == "RunnableSequence":
					try:
						chunk: str = dict(event_data_chunk[0])["log"]
						chunk = f"`[TOOL - CALLING]` {chunk}"
			
						await self._add_messages_to_history(
							history=history,
							history_type=history_type,
							msg_user=None,
							msg_ai=chunk,
						)
						result += chunk
						yield chunk
					except:
						pass
			
				elif event_name == "RunnableLambda":
					try:
						chunk = dict(event_data_chunk[1])["content"]
						chunk = f"`[TOOL - RESULT]` {chunk}\n\n"
			
						await self._add_messages_to_history(
							history=history,
							history_type=history_type,
							msg_user=None,
							msg_ai=chunk,
						)
			
						result += chunk
						yield chunk
					except:
						pass
						
		await self._add_messages_to_history(
			history=history,
			history_type=history_type,
			msg_user=input_message,
			msg_ai=result,
		)
	
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