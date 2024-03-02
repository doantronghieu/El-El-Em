from dotenv import load_dotenv

from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, \
                              MessagesPlaceholder
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory

import tools.sql as sql_tools
import tools.report as report_tools

from handlers.chat_model_start_handler import ChatModelStartHandler
################################################################################
load_dotenv()

tables = sql_tools.list_tables()

handler = ChatModelStartHandler()
chat = ChatOpenAI(callbacks=[handler])
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
tools = [sql_tools.run_query_tool, sql_tools.describe_tables_tool,
        report_tools.write_report_tool]
prompt = ChatPromptTemplate(
  messages=[
    SystemMessage(content=(
      'You are an AI that has access to a SQLite database.\n'
      f'The database has tables of: {tables}\n'
      "Do not make any assumptions about what tables exist or what columns "
      "exist. Instead, use the 'describe_tables' function"
      )),
    MessagesPlaceholder(variable_name='chat_history'), # added BEFORE the user input
    HumanMessagePromptTemplate.from_template('{input}'),
    MessagesPlaceholder(variable_name='agent_scratchpad')])
agent = OpenAIFunctionsAgent(llm=chat, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent, 
                              # verbose=True, 
                              tools=tools,
                              memory=memory)

# agent_executor('How many users are in the database?')
# agent_executor('How many users have provided a shipping address?')
# agent_executor(('Summarize the top 5 most popular products.'
#                 'Write the results to a report file.'))
agent_executor(('How many orders are there? '
                'Write the results to an html report.'))
agent_executor(('Repeat the exact same process for users.'))