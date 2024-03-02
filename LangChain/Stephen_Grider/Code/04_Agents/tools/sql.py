import sqlite3
from pydantic import BaseModel # from pydantic.v1 import BaseModel
from typing import List
from langchain.tools import Tool

conn = sqlite3.connect('db.sqlite')

def list_tables():
  c = conn.cursor()
  c.execute("SELECT name FROM sqlite_master WHERE type='table';")
  rows = c.fetchall()
  return '\n'.join(row[0] for row in rows if row[0] is not None)

def run_sqlite_query(query):
  # tool when chatgpt decide to use
  c = conn.cursor()
  try:
    c.execute(query)
    return c.fetchall()
  except sqlite3.OperationalError as err:
    return f'The following error occured: {str(err)}'

class RunQueryArgsSchema(BaseModel):
  query: str

run_query_tool = Tool.from_function(
  name='run_sqlite_query',
  description='Run a sqlite query', # when to run the tool
  func=run_sqlite_query,
  args_schema=RunQueryArgsSchema
)

def describe_tables(table_names):
  c = conn.cursor()
  tables = ', '.join(f"'{table}'" for table in table_names)
  rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' and name IN ({tables});")
  return '\n'.join(row[0] for row in rows if row[0] is not None)

class DescribeTablesArgsSchema(BaseModel):
  table_names: List[str]

describe_tables_tool = Tool.from_function(
  name='describe_tables',
  description='Given a list of table names, returns the schema of those table',
  func=describe_tables,
  args_schema=DescribeTablesArgsSchema
)