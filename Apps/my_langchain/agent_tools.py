
from langchain.tools import (
  BaseTool, StructuredTool, tool
)

from langchain_community.tools.tavily_search import (
  TavilySearchResults, TavilyAnswer,
)

from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

from langchain.pydantic_v1 import BaseModel, Field

wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=1, 
                                            doc_content_chars_max=100)
wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)