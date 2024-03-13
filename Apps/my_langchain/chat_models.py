from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

chat_openai = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")

chat_anthropic = ChatAnthropic(
  temperature=0, model_name="claude-3-opus-20240229"
)
