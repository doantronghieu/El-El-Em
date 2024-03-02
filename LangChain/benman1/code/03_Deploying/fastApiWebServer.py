from config import set_environment
set_environment()

from fastapi import FastAPI
from langchain import ConversationChain
from langchain.chat_models import ChatOpenAI

from lanarky import LangchainRouter
from starlette.requests import Request
from starlette.templating import Jinja2Templates

app = FastAPI()

chain = ConversationChain(
  llm = ChatOpenAI(
    model_name="gpt-3.5-turbo-0613",
    temperature=0,
    streaming=True  
  ),
  verbose=True,
)

templates = Jinja2Templates(directory='templates')

@app.get('/')
async def get(request: Request):
  return templates.TemplateResponse('index.html', {'request': request})

langchain_router = LangchainRouter(
  langchain_url='/chat', langchain_object=chain, streaming_mode=1
)
langchain_router.add_langchain_api_route(
  url='/chat_json', langchain_object=chain, streaming_mode=2
)
langchain_router.add_langchain_api_route(
    url='/ws', langchain_object=chain, streaming_mode=2
)

app.include_router(langchain_router)

# uvicorn fastApiWebServer:app --reload