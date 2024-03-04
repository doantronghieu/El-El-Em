from langchain.prompts.prompt import PromptTemplate

prompt_template = PromptTemplate

template_general = """\
The following is a friendly conversation between a human and an AI. \
The AI is talkative and provides lots of specific details from its context. \
If the AI does not know the answer to a question, it truthfully say it does not \
know.

Current conversation:
{history}
Human: {input}
AI Assistant:"""
prompt_general = prompt_template(
  input_variables=["history", "input"], template=template_general,
)