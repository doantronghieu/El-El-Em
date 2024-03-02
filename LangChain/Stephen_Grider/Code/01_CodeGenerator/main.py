from langchain.llms import openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
import argparse
from dotenv import load_dotenv

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str,
                    default='Return a list of numbers', help='Task to be performed')
parser.add_argument('--language', type=str,
                    default='python', help='Language to be used')
args = parser.parse_args()

llm = openai.OpenAI()

code_prompt = PromptTemplate(
    template="Write a very short {language} function that will {task}",
    input_variables=["language", "task"]
)

test_prompt = PromptTemplate(
    template="Write a test for the following {language} code:\n{code}",
    input_variables=["language", "code"]
)

code_chain = LLMChain(llm=llm, prompt=code_prompt, output_key='code')
test_chain = LLMChain(llm=llm,prompt=test_prompt,output_key='test')
chain = SequentialChain(chains=[code_chain, test_chain],
                        input_variables=["language", "task"],
                        output_variables=["code", "test"])

result = chain({"language": args.language, "task": args.task})

print(f'CODE:\n{result["code"]}\n')
print(f'{result["test"]}\n')
# python main.py --language javascript --task 'print hello'
