from llama_index.llms.openai import OpenAI

llm_openai_3_5 = OpenAI(
  model="gpt-3.5-turbo", temperature=0.0, request_timeout=60.0,
)

llm_openai_4 = OpenAI(
  model="gpt-4", temperature=0.0, request_timeout=0.0,
)