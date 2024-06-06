import add_packages
from operator import itemgetter
from toolkit import sql 
from typing import Any, Awaitable, Callable, Optional, Type, Union, Dict, List

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, tool
from langchain.tools import StructuredTool
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.embeddings import Embeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.language_models.chat_models import  BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, Runnable
from langchain_core.tools import ToolException, ValidationError
from langchain_core.vectorstores import VectorStore

#*------------------------------------------------------------------------------
def parse_retriever_input(params: Dict):
    return params["messages"][-1].content

#*------------------------------------------------------------------------------

async def invoke_chain(chain: Runnable, input, is_async: bool = False):
  if is_async:
    await chain.ainvoke(input)

  chain.invoke(input)

#*------------------------------------------------------------------------------


#*------------------------------------------------------------------------------
class InputChainSql(BaseModel):
  question: str = Field(description="user question, natural language, NOT sql query")

class Table(BaseModel):
  """Table in SQL database."""

  name: str = Field(description="Name of table in SQL database.")

class MySqlChain:
  def __init__(
    self,
    my_sql_db: sql.MySQLDatabase,
  
    llm: BaseChatModel,
    embeddings: Embeddings,
    vectorstore: VectorStore,
  
    proper_nouns: list[str],
    k_retriever_proper_nouns: int, # 4

    examples_questions_to_sql: list[str],
    k_few_shot_examples: int, # 5
    
    is_debug: bool = False,
    
    tool_name: str = None,
    tool_description: str = None,
    tool_metadata: Optional[Dict[str, Any]] = None,
    tool_tags: Optional[List[str]] = None,
  ) -> None:
    self.is_debug = is_debug
    
    self.my_sql_db = my_sql_db
  
    self.db = SQLDatabase.from_uri(self.my_sql_db.get_uri())
  
    self.llm = llm
    self.embeddings = embeddings

    self.vectorstore = vectorstore
    
    self.proper_nouns = self._process_proper_nouns(proper_nouns)
    
    self.vectorstore_nouns = self.vectorstore.from_texts(
      self.proper_nouns, self.embeddings,
    )
    self.retriever_nouns = self.vectorstore_nouns.as_retriever(
      search_kwargs={"k": k_retriever_proper_nouns}
    )
  
    self.tool_name = tool_name
    self.tool_description = tool_description
    self.tool_metadata = tool_metadata
    self.tool_tags = tool_tags
    
    self.context = self.db.get_context()
    self.table_info = self.context['table_info']
    self.table_schema = self.db.table_info
    self.table_schema_compact = self.table_schema.split("\n\n")[0]
    self.table_names = "\n".join(self.db.get_usable_table_names())
  
    self.prompt_tpl_get_tables = f"""\
    Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
    The tables are:

    {self.table_names}

    Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed.\
    """

    self.prompt_get_tables = ChatPromptTemplate.from_messages([
      ("system", self.prompt_tpl_get_tables),
      ("human", "{input}"),
    ])
    
    self.chain_get_tables = (
      {
        "input": itemgetter("question")
      }
      |	self.prompt_get_tables
      | self.llm.bind_tools([Table])
      | PydanticToolsParser(tools=[Table])
      | RunnableLambda(self.chain_process_get_tables)
    ).with_retry()
    #*-----------------------------------------------------------------------------

    self.chain_retrieve_proper_nouns = (
      itemgetter("question")
      | self.retriever_nouns
      | (lambda docs: "\n".join(doc.page_content for doc in docs))
    ).with_retry()

    #*-----------------------------------------------------------------------------
    self.example_selector = SemanticSimilarityExampleSelector.from_examples(
      examples=examples_questions_to_sql,
      embeddings=self.embeddings,
      vectorstore_cls=self.vectorstore,
      k=k_few_shot_examples,
      input_keys=["input"],
    )

    self.chain_get_examples = (
      {
        "input": itemgetter("question")
      }
      | RunnableLambda(self.example_selector.select_examples)
    ).with_retry()

    #*-----------------------------------------------------------------------------
    # Gen sql, check sql, filter nouns
    self.prompt_tpl_write_sql = """\
    You are a {dialect} expert. Given an input question, creat a syntactically correct {dialect} query to run.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per {dialect}. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use date('now') function to get the current date, if the question involves "today".

    Here is the relevant table info:
    {table_info}

    Here is a non-exhaustive list of possible feature values. If filtering on a feature
    value make sure to check its spelling against this list first:
    {proper_nouns}

    Below are a number of examples of questions and their corresponding SQL queries.
    {examples}

    Write an initial draft of the query. Then double check the {dialect} query for common mistakes, including:
    - Using NOT IN with NULL values
    - Using UNION when UNION ALL should have been used
    - Using BETWEEN for exclusive ranges
    - Data type mismatch in predicates
    - Properly quoting identifiers
    - Using the correct number of arguments for functions
    - Casting to the correct data type
    - Using the proper columns for joins

    Use format:

    First draft: <<FIRST_DRAFT_QUERY>>
    Final answer: <<FINAL_ANSWER_QUERY>>\
    """
    self.prompt_write_sql = ChatPromptTemplate.from_messages([
      ("system", self.prompt_tpl_write_sql),
      ("human", "{input}"),
    ]).partial(dialect=self.db.dialect)

    self.chain_write_sql = (
      create_sql_query_chain(
        self.llm, self.db, prompt=self.prompt_write_sql
      )
      | RunnableLambda(self.chain_process_parse_final_answer)	
    ).with_retry()

    #*-----------------------------------------------------------------------------

    self.query_executor = QuerySQLDataBaseTool(db=self.db)

    self.prompt_answer = PromptTemplate.from_template("""\
    Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer:\
    """)

    self.chain_answer = (
      self.prompt_answer | self.llm | StrOutputParser()
    ).with_retry()

    self.chain_sql = (
      RunnablePassthrough
      .assign(table_names_to_use=self.chain_get_tables)
      .assign(proper_nouns=self.chain_retrieve_proper_nouns)
      .assign(examples=self.chain_get_examples)
      .assign(query=self.chain_write_sql)
      .assign(result=itemgetter("query") | self.query_executor)
      .assign(output=self.chain_answer)
    ).with_retry()
    self.metadata_chain_sql = {"is_my_sql_chain_run": True}

  def _process_proper_nouns(self, proper_nouns: list):
    result = proper_nouns
    
    for i in range(len(result)):
      if type(result[i]) != str:
        key = next(iter(result[i]))
        value = result[i][key]
        result[i] = f'{key}-{value}'
    
    return result

  def chain_process_get_tables(self, tables: list):
    result = [table.name for table in tables]
    return result

  def chain_process_parse_final_answer(self, output: str) -> str:
    return output.split("Final answer: ")[1]

  def prepare_chain_input(self, question: str):
    result = {
      "question": question,
    }
    return result

  def invoke_chain(self, question: str) -> Union[str, dict]:
    """Get natural user question, turn it into SQL query and execute"""
    question = self.prepare_chain_input(question)
    result = self.chain_sql.invoke(
      question,
      config={"metadata": self.metadata_chain_sql},
    )
    
    if not self.is_debug:
      result = result["output"]

    return result
  
  async def ainvoke_chain(self, question: str) -> Union[str, dict]:
    """Get natural user question, turn it into SQL query and execute"""
    question = self.prepare_chain_input(question)
    result = await self.chain_sql.ainvoke(
      question,
      config={"metadata": self.metadata_chain_sql},
    )
    
    if not self.is_debug:
      result = result["output"]

    return result
  
  def create_tool_chain_sql(
    self,
    func: Callable = None,
    args_schema: Type[BaseModel] = None,
    coroutine: Optional[Callable[..., Awaitable[Any]]] = None,
    name: str = None,
    description: str = None,
    return_direct: bool = False, # True: Agent will stop after tool completed
    handle_tool_error: Optional[Union[bool, str, Callable[[ToolException], str]]] = True,
    handle_validation_error: Optional[Union[bool, str, Callable[[ValidationError], str]]] = True,
    verbose: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
  ):
    func = self.invoke_chain if func is None else func
    args_schema = InputChainSql if args_schema is None else args_schema
    coroutine = self.ainvoke_chain if coroutine is None else coroutine
    
    name = self.tool_name if name is None else name
    description = self.tool_description if description is None else description
    metadata = self.tool_metadata if metadata is None else metadata
    tags = self.tool_tags if tags is None else tags
    
    tool_chain_sql = StructuredTool.from_function(
      func=func,
      args_schema=args_schema,
      coroutine=coroutine,
      name=name,
      description=description,
      return_direct=return_direct,
      handle_tool_error=handle_tool_error,
      handle_validation_error=handle_validation_error,
      verbose=verbose,
      metadata=metadata,
      tags=tags,
    )

    return tool_chain_sql

class MyRoutableChain:
    def __init__(
      self, 
      categories: list[str], 
      chains: dict[str, Runnable],
      llm: BaseChatModel,
    ):
        self.llm = llm
      
        self.categories = categories
        self.chains = chains
        
        self.prompt_tpl_extract_category = self.create_prompt_tpl_extract_category()
        self.prompt_extract_category = PromptTemplate.from_template(self.prompt_tpl_extract_category)
        
        self.chain_extract_category = (
            self.prompt_extract_category
            | self.llm
            | StrOutputParser()
        )
        
        self.chain_routable = (
            {
                "question": lambda x: x["question"],
                "category": self.chain_extract_category,
            }
            | RunnableLambda(self.route_chains)
        )

    def create_prompt_tpl_extract_category(self):
        categories = ", ".join([f"{cate}" for cate in self.categories])
        prompt_tpl_extract_category = """\
        You will be given a list of categories and a question. Your task is to determine which category the question falls into and respond with only the name of that category. Do not include any other words in your response.

        Here is the list of categories:
        <categories>
        {{CATEGORIES}}
        </categories>

        Here is the question to classify:
        <question>
        {{QUESTION}}
        </question>

        <example>
        <categories>
        science, history, math, literature
        </categories>

        <question>
        What is the chemical formula for water?
        </question>

        science
        </example>

        Now classify the provided question into one of the given categories. Remember, respond with only a single word - the name of the category.

        <question>
        {question}
        </question>

        Classification:
        """

        prompt_tpl_extract_category = prompt_tpl_extract_category.replace("{{CATEGORIES}}", categories)
        return prompt_tpl_extract_category

    def route_chains(self, chain_vars):
        for cat, chain in self.chains.items():
            if cat.lower() == chain_vars["category"]:
                return chain
    
    def get_chain(self):
        return self.chain_routable


...