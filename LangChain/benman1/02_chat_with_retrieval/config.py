import os

OPENAI_API_KEY = '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'
HUGGINGFACEHUB_API_TOKEN = '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'
REPLICATE_API_TOKEN = 'r8_1mJX0C8gULAaN57NPYfhtYEDpEJIgI13zxzoh'
WOLFRAM_ALPHA_APPID = 'U6KWKU-U4L6UKQPXL'


def set_environment():
    variable_dict = globals().items()
    for key, value in variable_dict:
        if "API" in key or "ID" in key:
            os.environ[key] = value
