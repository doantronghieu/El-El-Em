import os

OPENAI_API_KEY = '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'
HUGGINGFACEHUB_API_TOKEN = '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'
REPLICATE_API_TOKEN = 'r8_1mJX0C8gULAaN57NPYfhtYEDpEJIgI13zxzoh'
WOLFRAM_ALPHA_APPID = 'U6KWKU-U4L6UKQPXL'
PROMPTWATCH_API_KEY = "Wlo5WFgyaFVlMmFMaWsxejZsSnZmMDZSbEhIMjozMWRjZjQ3NC03MGRhLTUxNWMtYTQyNC1iMjExMjQ4YjZkZTg="

LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_API_KEY = "ls__b042636c26cf4c6bb4f9d0eb542943d0"

TAVILY_API_KEY = "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"

PINECONE_API_KEY = "4c183128-9d40-4846-9620-30f636051544"
PINECONE_ENVIRONMENT = "gcp-starter"


def set_environment():
    variable_dict = globals().items()
    for key, value in variable_dict:
        if "API" in key or "ID" in key:
            os.environ[key] = value


# Initialize environment variables when this module is imported
set_environment()
