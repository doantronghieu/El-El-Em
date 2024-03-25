import typing

from langchain.chains import (
    LLMChain, ConversationChain, RetrievalQA
)
from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)

#*------------------------------------------------------------------------------
def parse_retriever_input(params: typing.Dict):
    return params["messages"][-1].content

#*------------------------------------------------------------------------------
