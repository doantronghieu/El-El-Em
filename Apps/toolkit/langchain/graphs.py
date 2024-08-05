from loguru import logger

from typing import (
  Annotated as t_Annotated, Literal as t_Literal
)
from typing_extensions import TypedDict as t_TypedDict
from langchain_core.pydantic_v1 import BaseModel as p_BaseModel

from operator import (
  add as o_add,
)

from langgraph.graph.message import add_messages as o_add_messages

from langgraph.graph import (
  START, END, MessageGraph, StateGraph, Graph, MessagesState
)
from langgraph.graph.graph import CompiledGraph

from langgraph.prebuilt import (
  ToolExecutor, ToolInvocation, chat_agent_executor, create_agent_executor,
  ToolNode, tools_condition,
)


from langgraph.checkpoint.memory import MemorySaver as cp_MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver as cp_BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver as cp_SqliteSaver

def display_graph(graph: CompiledGraph):
  from IPython.display import Image, display
  try:
    display(Image(graph.get_graph().draw_mermaid_png()))
  except Exception:
    logger.error("Error displaying graph.")
    pass