import add_packages
import config
from loguru import logger

import chainlit as cl
from chainlit.types import ThreadDict

from my_langchain import (
  chat_models, prompts, output_parsers, chains
)

#* =============================================================================



model = chat_models.chat_openai
prompt = prompts.ChatPromptTemplate.from_messages([
  (
    "system",
    ("You're a very knowledgeable historian who provides accurate and eloquent "
     "answers to historical questions."),
  ),
  (
    "human",
    "{question}",
  )
])
chain = chains.LLMChain(
  llm=model, prompt=prompt, output_parser=output_parsers.StrOutputParser()
)
user_session = "chain"

#* =============================================================================
@cl.on_chat_start
# A hook called when a new chat session is created.
async def on_chat_start():
  # Whenever user connects to Chainlit app, new chat session is created. 
  # Chat session goes through life cycle of events, respond by defining hooks.
  # Creates a runnable for each chat session.
  
  logger.info("A new chat session has started.")
  
  cl.user_session.set(user_session, chain)
  
# Incoming messages from the UI.
@cl.on_message
# A hook called when a new message is received from the user.
async def on_message(msg: cl.Message):
  # The response is generated whenever a user sends a message.
  chain = cl.user_session.get(user_session)
  
  # The callback handler listens to chain's steps and sends them to the UI.
  res = await chain.arun(
    question=msg.content, callbacks=[cl.LangchainCallbackHandler()]
  )
  
  logger.info(f"The user sent: {msg.content}")
  
  await cl.Message(content=res).send()

@cl.on_stop
# A hook triggered when the stop button is clicked during a running task.
async def on_stop():
  logger.info("The user wants to stop the task.")

@cl.on_chat_end
# A hook called when the chat session ends due to user disconnection or 
# starting a new session.
async def on_chat_end():
  logger.info("The user disconnected.")

@cl.on_chat_resume
# A hook called when a user resumes a chat session after disconnection, only if 
# authentication and data persistence are enabled.
async def on_chat_resume(thread: ThreadDict):
  logger.info("The user resumed a previous chat session.")

# - The -w flag enables auto-reloading in Chainlit, 
# chainlit run app_chainlit.py -w