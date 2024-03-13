from langchain.agents import tool

from langchain_community.tools.tavily_search import (
  TavilySearchResults, TavilyAnswer,
)

from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper