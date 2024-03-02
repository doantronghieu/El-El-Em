from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import langchain_openai
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory, FileChatMessageHistory
from langchain.chains import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate, HumanMessagePromptTemplate, ChatMessagePromptTemplate

from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
###############################################################################
load_dotenv()

app = FastAPI()

chat = langchain_openai.ChatOpenAI()
memory = ConversationBufferMemory(
    memory_key='messages', return_messages=True,
    chat_memory=FileChatMessageHistory('messages.json'))

prompt = ChatPromptTemplate(
    input_variables=['content', 'messages'],
    messages=[
        MessagesPlaceholder(variable_name='messages'),
        HumanMessagePromptTemplate.from_template('{content}')
    ]
)

chain = LLMChain(llm=chat, prompt=prompt, memory=memory)
###############################################################################


class UserInput(BaseModel):
    content: str
###############################################################################

# Local function to answer the question


def provide_services():
    services = ["General cleaning", "Specialized cleaning"]
    print("Local function called: Services provided")
    return services


def llm(content):
    result = chain({'content': content})
    return result['text']

###############################################################################


@app.post("/chatbot/")
async def chatbot_response(user_input: UserInput):
    print('Function called')
    if user_input.content.strip().lower() == "what services do you provide?":
        response = provide_services()
    else:
        response = llm(user_input.content)

    return {"response": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# uvicorn chatbot_service:app --host 0.0.0.0 --port 8000 --reload
