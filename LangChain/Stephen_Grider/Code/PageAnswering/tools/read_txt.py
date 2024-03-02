from pydantic import BaseModel
from typing import List
from langchain.tools import Tool


def read_txt(filename):
    with open(filename, 'r', encoding="utf-8") as f:
        return f.read()

class ReadTxtArgsSchema(BaseModel):
    filename: str

read_txt_tool = Tool.from_function(
    name='read_txt',
    description='Given a filename, return the content of the file',
    func=read_txt,
    args_schema=ReadTxtArgsSchema
)
