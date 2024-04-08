from llama_index.llms.openai import OpenAI

openai_llms = {
  "GPT-3.5-TURBO": OpenAI(
    model="gpt-3.5-turbo", temperature=0.0, request_timeout=60.0,
  ),
  "GPT-4": OpenAI(
    model="gpt-4", temperature=0.0, request_timeout=60.0,
  ),
}
