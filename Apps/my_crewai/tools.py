import add_packages
import json, os, requests
from crewai import Agent, Task
from my_langchain.agent_tools import tool
from unstructured.partition.html import partition_html

class Browser():
  @tool("Scrape website content")
  def scrape_and_summarize_website(url: str):
    """Useful to scrape and summarize a website content"""
    pass