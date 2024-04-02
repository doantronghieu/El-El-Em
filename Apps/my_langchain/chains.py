import typing

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.llm import LLMChain
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)

from langchain_core.runnables import Runnable
from langchain.chains.query_constructor.schema import AttributeInfo

#*------------------------------------------------------------------------------
def parse_retriever_input(params: typing.Dict):
    return params["messages"][-1].content

#*------------------------------------------------------------------------------

async def invoke_chain(chain: Runnable, input, is_async: bool = False):
  if is_async:
    await chain.ainvoke(input)

  chain.invoke(input)