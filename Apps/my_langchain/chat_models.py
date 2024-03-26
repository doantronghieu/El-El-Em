import config
import os
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatCohere

chat_openai = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")

chat_anthropic = ChatAnthropic(
  temperature=0, model_name="claude-3-opus-20240229"
)

chat_cohere = ChatCohere(cohere_api_key=os.getenv("COHERE_API_KEY"))
