
from langchain.tools import (
  BaseTool, StructuredTool, tool, 
)

from langchain_core.tools import (
  ToolException,
)

import langchain_community.tools as tools_community

from langchain_community.tools.tavily_search import (
  TavilySearchResults, TavilyAnswer,
)
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.tools.e2b_data_analysis.tool import E2BDataAnalysisTool
from langchain_community.tools.file_management import MoveFileTool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

from langchain.pydantic_v1 import BaseModel, Field

from langchain.callbacks.manager import (
  AsyncCallbackManagerForToolRun, CallbackManagerForToolRun,
)

from langchain_core.utils.function_calling import (
  convert_to_openai_function
)

import typing

wikipedia_api_wrapper = WikipediaAPIWrapper(top_k_results=1, 
                                            doc_content_chars_max=100)
wikipedia = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)


def _handle_error(error: ToolException) -> str:
  return (
    "The following errors occurred during tool execution:"
    + error.args[0]
    + "Please try another tool."
  )
  