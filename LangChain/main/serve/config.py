import os

OPENAI_API_KEY = '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'
HUGGINGFACEHUB_API_TOKEN = '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'
REPLICATE_API_TOKEN = 'r8_1mJX0C8gULAaN57NPYfhtYEDpEJIgI13zxzoh'
WOLFRAM_ALPHA_APPID = 'U6KWKU-U4L6UKQPXL'
PROMPTWATCH_API_KEY = "Wlo5WFgyaFVlMmFMaWsxejZsSnZmMDZSbEhIMjozMWRjZjQ3NC03MGRhLTUxNWMtYTQyNC1iMjExMjQ4YjZkZTg="
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_API_KEY = "ls__b042636c26cf4c6bb4f9d0eb542943d0"
TAVILY_API_KEY = "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"

def set_environment():
    variable_dict = globals().items()
    for key, value in variable_dict:
        if "API" in key or "ID" in key:
            os.environ[key] = value
