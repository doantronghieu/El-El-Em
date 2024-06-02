from unittest import result
import add_packages
from operator import itemgetter
from toolkit import sql 
import typing
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain.tools import tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_core.embeddings import Embeddings
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.language_models.chat_models import  BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, Runnable
from langchain_core.vectorstores import VectorStore


#*------------------------------------------------------------------------------
def parse_retriever_input(params: typing.Dict):
    return params["messages"][-1].content

#*------------------------------------------------------------------------------

async def invoke_chain(chain: Runnable, input, is_async: bool = False):
  if is_async:
    await chain.ainvoke(input)

  chain.invoke(input)



#*------------------------------------------------------------------------------
class Table(BaseModel):
  """Table in SQL database."""

  name: str = Field(description="Name of table in SQL database.")

class InputChainSql(BaseModel):
  user_input: str = Field(description="user question about data")

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
    
    is_debug: bool = True,
  ) -> None:
    self.is_debug = is_debug
    
    self.my_sql_db = my_sql_db
  
    self.db = SQLDatabase.from_uri(self.my_sql_db.get_uri())
  
    self.llm = llm
    self.embeddings = embeddings

    self.vectorstore = vectorstore
    self.vectorstore_nouns = self.vectorstore.from_texts(
      proper_nouns, self.embeddings,
    )
    self.retriever_nouns = self.vectorstore_nouns.as_retriever(
      search_kwargs={"k": k_retriever_proper_nouns}
    )
  
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
    )
    #*-----------------------------------------------------------------------------

    self.chain_retrieve_proper_nouns = (
      itemgetter("question")
      | self.retriever_nouns
      | (lambda docs: "\n".join(doc.page_content for doc in docs))
    )

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
    )

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
    )

    #*-----------------------------------------------------------------------------

    self.query_executor = QuerySQLDataBaseTool(db=self.db)

    self.prompt_answer = PromptTemplate.from_template("""\
    Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer:\
    """)

    self.chain_answer = self.prompt_answer | self.llm | StrOutputParser()

    self.chain_sql = (
      RunnablePassthrough
      .assign(table_names_to_use=self.chain_get_tables)
      .assign(proper_nouns=self.chain_retrieve_proper_nouns)
      .assign(examples=self.chain_get_examples)
      .assign(query=self.chain_write_sql)
      .assign(result=itemgetter("query") | self.query_executor)
      .assign(output=self.chain_answer)
    ).with_retry()

    self.tool_chain_sql = StructuredTool.from_function(
      func=self.invoke_chain,
      name="SQL executor",
      description="generate sql query base on user question and execute sql query"
    )
    
  def chain_process_get_tables(self, tables: list):
    result = [table.name for table in tables]
    return result

  def chain_process_parse_final_answer(self, output: str) -> str:
    return output.split("Final answer: ")[1]

  def prepare_chain_input(self, user_input: str):
    result = {
      "question": user_input,
    }
    return result

  
  def invoke_chain(self, user_input: str):
    """Get natural user question, turn it into SQL query and execute"""
    user_input = self.prepare_chain_input(user_input)
    result = self.chain_sql.invoke(user_input)
    
    if not self.is_debug:
      result = result["output"]

    return result