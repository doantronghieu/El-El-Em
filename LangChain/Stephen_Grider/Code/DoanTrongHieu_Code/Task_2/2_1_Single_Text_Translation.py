# Single Text Translation

from langchain.llms import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import argparse
from dotenv import load_dotenv
from pyboxen import boxen
################################################################################

load_dotenv()

def boxen_print(*args, **kwargs):
    print(boxen(*args, **kwargs))

llm = openai.OpenAI()

language_dict = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'vi': 'Vietnamese',
    # Add more language codes and names as needed
}

################################################################################

code_prompt = PromptTemplate(
    template='Translate "{text}" to {dest_language}',
    input_variables=["text", "dest_language"]
)

code_chain = LLMChain(
    llm=llm,
    prompt=code_prompt
)


while True:
    dest_language = input('>> Destination language: ')
    text = input('>> Text: ')
    
    translated_language = language_dict.get(dest_language, 'Unknown')

    result = code_chain(
        {"text": text,
         "dest_language": translated_language}
    )

    result['text'] = result['text'].replace('\n\n', '')
    
    boxen_print(result['text'], title='Translation', color='green')

# python 2_1_Single_Text_Translation.py