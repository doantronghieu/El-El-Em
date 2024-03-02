import argparse
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import openai
from dotenv import load_dotenv
from pyboxen import boxen

from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate, \
    HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.agents import OpenAIFunctionsAgent, AgentExecutor
from langchain.schema import SystemMessage
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory, FileChatMessageHistory

import tools.scraping as scraping_tools
import tools.read_txt as read_txt_tools

from handlers.chat_model_start_handler import ChatModelStartHandler
################################################################################
load_dotenv()


def boxen_print(*args, **kwargs):
  print(boxen(*args, **kwargs))
################################################################################

tools = [scraping_tools.extract_website_content_tool,
         read_txt_tools.read_txt_tool]

handler = ChatModelStartHandler()
chat = ChatOpenAI(callbacks=[handler])
memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True)
prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(content=(
            "You are an AI chatbot that can answer user questions about the "
            "company website. The chatbot will do this by first crawling the "
            "website to gather relevant data, then preprocessing and "
            "structuring this data into a searchable index. Once the index is "
            "created, the chatbot will use natural language processing to "
            "comprehend user questions and search the index for the best match."
            "The chatbot will then deliver the most accurate response drawn "
            "from the website's information."
        )),
        # added BEFORE the user input
        MessagesPlaceholder(variable_name='chat_history'),
        HumanMessagePromptTemplate.from_template('{input}'),
        MessagesPlaceholder(variable_name='agent_scratchpad')])
agent = OpenAIFunctionsAgent(llm=chat, prompt=prompt, tools=tools)
agent_executor = AgentExecutor(agent=agent,
                               # verbose=True,
                               tools=tools,
                               memory=memory)

url = "https://www.presight.io/privacy-policy.html"
agent_executor(
    f'Crawl content of this website: "{url}"')
################################################################################
data = read_txt_tools.read_txt('company_info.txt')
llm = openai.OpenAI()
chat = ChatOpenAI()

"""
Example questions:
How can I contact you?
How is the Data Security of your company?
What is the purpose of Use of Data?

Trần Anh Tuấn là ai?
"""

################################################################################
question_prompt_template = PromptTemplate(
    template=(f"Based on the data provided:\n\n{data}"
              'Answer the question: {question}'),
    input_variables=["data", "question"]
)

question_chain = LLMChain(
    llm=llm, prompt=question_prompt_template, output_key='answer')

while True:
    question = input('>> ')
    question_result = question_chain({"data": data, "question": question})
    boxen_print(question_result['answer'].split('\n\n')[
                1], title='AI bot', color='green')
