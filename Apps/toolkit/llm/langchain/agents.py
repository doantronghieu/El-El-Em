
import add_packages
import boto3
import os
from operator import itemgetter
from loguru import logger
from typing import Union, Optional, List, Literal, AsyncGenerator, TypeAlias
from pydantic import BaseModel

from toolkit import utils

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, Runnable, RunnableParallel

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

from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories.dynamodb import DynamoDBChatMessageHistory
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
#*==============================================================================

dynamodb = boto3.resource("dynamodb")

#*==============================================================================

TypeAgent: TypeAlias = Literal["tool_calling", "openai_tools", "react", "anthropic"]

TypeHistoryType: TypeAlias = Literal["in_memory", "dynamodb", "mongodb"]
TypeUserId: TypeAlias = Optional[str]

TypeSessionId: TypeAlias = Union[str, None]

#*==============================================================================

prompt_tpl_check_ans = """\
You are tasked with evaluating whether an AI's answer adequately addressed and satisfied the original question. Follow these steps carefully:

1. Review the following:

Original question:
<original_question>
{original_question}
</original_question>

AI's answer:
<ai_answer>
{ai_answer}
</ai_answer>

2. Analyze the AI's answer by considering the following:
   - Does the answer directly address the main points of the original question?
   - Is the information provided relevant and accurate?
   - Does the answer provide sufficient detail to satisfy the query?
   - Are there any parts of the original question left unanswered?
   - Ensure that all keywords in the answer correspond to the keywords in the question.

3. Based on your analysis, determine if the AI's answer adequately addressed and satisfied the original question.

4. Provide your response as follows:
   - If the answer adequately addressed and satisfied the query, output exactly: True
   - If the answer did not adequately address or satisfy the query, or if you cannot determine this due to lack of information or context, output exactly: ERROR

Do not provide any explanation or justification for your response. Your output must be either "True" or "ERROR" without any additional text.

Examples of correct outputs:
True
ERROR

Ensure your response is only one of these two options.
"""
prompt_check_ans = ChatPromptTemplate.from_template(prompt_tpl_check_ans)

def check_ans(original_question: str, ai_answer: str, llm=ChatOpenAI()):
	chain_check_ans = (
			{
				"original_question": itemgetter("original_question"),
				"ai_answer": itemgetter("ai_answer")
			}
			| prompt_check_ans
			| llm
			| StrOutputParser()
	).with_retry()

	result = chain_check_ans.invoke({
		"original_question": original_question,
		"ai_answer": ai_answer,
	})
	return result

#*------------------------------------------------------------------------------

prompt_tpl_res_if_not_satis = """\
You are tasked with generating a response to a user based on the number of retries for an AI-generated answer. You will be given three inputs: the AI's answer, the current retry count, and the maximum number of retries allowed.

Here are the inputs you will work with:

<ai_answer>
{ai_answer}
</ai_answer>

<current_retry>{current_retry}</current_retry>

<max_retry>{max_retry}</max_retry>

Follow these steps to generate the appropriate response:

1. Compare the value of current_retry to max_retry.

2. If current_retry is less or equals to than max_retry:
   - Inform the user that you will continue to retry to get the correct answer.
   
3. If current_retry is greater than max_retry:
   - Tell the user to please try again.

4. Ensure that your response is in the exact same language as the text in ai_answer. This means you should analyze the language used in ai_answer and formulate your response in that same language.

Remember, do not include any explanations or additional information. Your output should only be the appropriate message to the user, written in the same language as ai_answer.

Your response: \
"""
prompt_res_if_not_satis = ChatPromptTemplate.from_template(prompt_tpl_res_if_not_satis)


def response_if_not_satisfied(
  ai_answer: str, current_retry: int, max_retry: int,
  llm=ChatOpenAI()
):
	chain_res_if_not_satis = (
			{
				"ai_answer": itemgetter("ai_answer"),
				"current_retry": itemgetter("current_retry"),
				"max_retry": itemgetter("max_retry")
			}
			| prompt_res_if_not_satis
			| llm
			| StrOutputParser()
	).with_retry()
 
	result = chain_res_if_not_satis.invoke({
		"ai_answer": ai_answer,
		"current_retry": current_retry,
		"max_retry": max_retry,
	})
	return result

#*==============================================================================

class SchemaChatHistory(BaseModel):
	history_type: TypeHistoryType = "in_memory"
	user_id: TypeUserId = "admin"
	session_id: TypeSessionId = None
	history_size: Union[int, None] = 10

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
		elif self.history_type == "mongodb":
			self.chat_history = MongoDBChatMessageHistory(
				session_id=self.session_id, # user name, email, chat id etc.
				connection_string=os.getenv("MONGODB_ATLAS_CLUSTER_URI"),
				database_name=os.getenv("MONGODB_DB_NAME"),
				collection_name=os.getenv("MONGODB_COLLECTION_NAME_MSG"),
			)			

		if self.is_new_session:
			welcome_msg = "Hello! How can I help you today?"
			if self.history_type == "in_memory":
				self.chat_history.append(AIMessage(welcome_msg))
			elif self.history_type == "dynamodb" or self.history_type == "mongodb":
				self.chat_history.add_ai_message(welcome_msg)

		if self.user_id: logger.info(f"User Id: {self.user_id}")
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
		elif self.history_type == "dynamodb" or self.history_type == "mongodb":
			if msg_user:
				await self.chat_history.aadd_messages(messages=[HumanMessage(msg_user)])
			if msg_ai:
				await self.chat_history.aadd_messages(messages=[AIMessage(msg_ai)])

	async def _get_chat_history(self, is_truncate=True):
		if self.history_type == "in_memory":
			result = self.chat_history
			if is_truncate: result = result[-self.history_size:]
		elif self.history_type == "dynamodb":
			result = self.chat_history.messages
		elif self.history_type == "mongodb":
			result = await self.chat_history.aget_messages()
			if is_truncate: result = result[-self.history_size:]

		return result

	async def clear_chat_history(self):
		if self.history_type == "in_memory":
			self.chat_history = []
		elif self.history_type == "dynamodb" or self.history_type == "dynamodb":
			await self.chat_history.aclear()

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
		history_type: TypeHistoryType = "mongodb",
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
	
	async def invoke_agent(
		self,
		input_message: str,
		callbacks: Optional[List] = None,
		mode: Literal["sync", "async"] = "async",
	
		history_type: TypeHistoryType = "mongodb",
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

	async def astream_events_basic_wrapper(
		self,
		input_message: str,
	):
		result = ""
		async for chunk in self.astream_events_basic(input_message):
			result += chunk
			print(chunk, end="", flush=True)
		return result

	async def astream_events_basic(
		self,
		input_message: str,

		history_type: TypeHistoryType = "mongodb",
		user_id: TypeUserId = utils.generate_unique_id(thing="uuid_name"),
		session_id: TypeSessionId = utils.generate_unique_id(thing="uuid"),
	
		show_tool_call: bool = False,
		history_size: Union[int, None] = 10,
	) -> AsyncGenerator[str, None]:
		"""
		async for chunk in agent.astream_events_basic("Hello"):
			print(chunk, end="", flush=True)
		"""

		history = self._create_chat_history(
			history_type, user_id, session_id, history_size,
		)

		result = ""
		is_result_satisfied = False
		max_retry = 1
		current_retry = 0

		""" used for debugging
		a = agent.events
		a = [x for x in agent.events if x["event"] == "on_chat_model_stream"]
		a_data = [x["data"] for x in a]
		a_data_chunk = [x["chunk"] for x in a_data]
		a_data_chunk_tool = [x for x in a if dict(a_data_chunk)["tool_call_chunks"]]
		a_metadata_sql_chain = [x for x in a if "..." in x["metadata"].keys()]
		"""
  
		# self.events = [] # debug

		while (not is_result_satisfied) and (current_retry <= max_retry):
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
							or ("is_my_sql_chain_run" in event.get("metadata", {})) \
							or ("is_my_rag_chain_run" in event.get("metadata", {})):
						continue

					result += chunk
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
			
			current_retry += 1

			if check_ans(
     		original_question=input_message, ai_answer=result, llm=self.llm,
      ) != "ERROR":
				is_result_satisfied = True
				yield("\n")
			else:
				if current_retry <= max_retry:
					for _ in range(1):
						yield("\n\n")
			
					for res in response_if_not_satisfied(
						ai_answer=result,
						current_retry=current_retry,
						max_retry=max_retry,
						llm=self.llm,
					):
						yield res
					for _ in range(1):
						yield("\n\n")
					
					history = self._create_chat_history(
						history_type=history_type,
						user_id=utils.generate_unique_id(thing="uuid_name"), 
						session_id=utils.generate_unique_id(thing="uuid"), 
						history_size=history_size,
					)

				result = ""
    
		await self._add_messages_to_history(
			history=history,
			history_type=history_type,
			msg_user=input_message,
			msg_ai=result,
		)
	
def hello():
	...