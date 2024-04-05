import add_packages
import yaml, re, os, json, requests
from crewai import Agent, Task
from crewai import Agent, Task
from pprint import pprint
from bs4 import BeautifulSoup
from my_langchain.agent_tools import tool
from my_langchain import (
  document_loaders, text_splitters, documents, agent_tools
)

#*==============================================================================
with open(f"{add_packages.APP_PATH}/my_configs/crew_ai.yaml") as f:
  configs_crewai = yaml.safe_load(f)

#*==============================================================================
def process_scraped_content(text: str) -> str:
  # Replace "\n " with "\n"
  text = re.sub(r'\n\s*', '\n', text)

  # Replace consecutive newlines with a single newline
  text = re.sub(r'\n+', '\n', text)

  # Replace multiple spaces with a single space
  text = re.sub(r' +', ' ', text)

  # Split text into lines
  lines = text.split('\n')

  # Filter out lines with fewer than 4 words
  lines = [line for line in lines if len(line.split()) >= 4]

  # Remove words starting with "Ä"
  lines = [re.sub(r'\bÄ\w*\b', '', line) for line in lines]

  # List of phrases to remove lines containing them
  phrases_to_remove = [
      "Sign in to your CNN account",
      "Ad was repetitive to ads I've seen previously",
      "Content moved around while ad loaded"
  ]

  # Filter out lines containing any of the phrases
  lines = [line for line in lines if not any(
      phrase in line for phrase in phrases_to_remove)]

  # Join filtered lines back into text
  text = '\n'.join(lines)

  return text

#*==============================================================================
class ToolsBrowser():
  
  @tool("Scrape website content")
  def scrape_and_summarize_website(url: str) -> str:
    """Useful to scrape and summarize a website content"""
    
    doc = document_loaders.WebBaseLoader(url).load()[0].page_content
    doc = process_scraped_content(doc)
    doc = [documents.Document(doc)]

    text_splitter = text_splitters.RecursiveCharacterTextSplitter(
      chunk_size=2000, chunk_overlap=300,
    )
    docs = text_splitter.split_documents(doc)
    
    configs_tool_browser = configs_crewai["tools"]["browser"]
    configs_tool_scrape_and_summarize_website = configs_tool_browser["scrape_and_summarize_website"]
    configs_tool_scrape_and_summarize_website_agent = configs_tool_scrape_and_summarize_website["agent"]
    configs_tool_scrape_and_summarize_website_task = configs_tool_scrape_and_summarize_website["task"]
    
    agent = Agent(
      **configs_tool_scrape_and_summarize_website_agent,
      allow_delegation=False,
    )

    summaries = []
    for doc in docs:
      content = doc.page_content
      description = configs_tool_scrape_and_summarize_website_task["description"].replace(
        "%CHUNK%", content)
      expected_output = configs_tool_scrape_and_summarize_website_task["expected_output"]

      task = Task(
        agent=agent,
        description=description,
        expected_output=expected_output,
      )

      summary = task.execute()
      summaries.append(summary)

    return "\n\n".join(summaries)

  @tool("Search the internet")
  def search_serper(
    query: str,
    top_result_to_return: int = 4,
  ) -> str:
    """
    Useful to search the internet about a a given topic and return relevant 
    results
    """
    url_serper = "https://google.serper.dev/search"
    
    payload = json.dumps({"q": query})
    headers = {
      "X-API-KEY": os.environ["SERPER_API_KEY"],
      "content-type": "application/json",
    }

    response = requests.request("POST", url_serper, headers=headers, data=payload)
    
    if "organic" not in response.json():
      return ("Sorry, I couldn't find anything about that, there could be an "
              "error with you serper api key.")
    else:
      results = response.json()["organic"]
      result_str = []
      
      for result in results[:top_result_to_return]:
        try:
          result_str.append("\n".join([
            f"Title: {result['title']}", f"Link: {result['link']}",
            f"Snippet: {result['snippet']}", f"\n{'-'*10}"
          ]))
        except KeyError:
          next
    
    return "\n".join(result_str)

  search_duckduckgo = agent_tools.DuckDuckGoSearchRun()
  
#*==============================================================================
class ToolsCalculator():
  
  @tool("Make a calculation")
  def calculate(operation):
    """
    Useful to perform any mathematical calculations, like sum, minus, multiplication,
    division, etc. 
    The input to this tool should be mathematical expression, a couple of examples
    are `200*7` or `5000/2*10`
    """
    try:
      return eval(operation)
    except SyntaxError:
      return "Error: Invalid syntax in mathematical expression"
  
  python_repl = agent_tools.Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=agent_tools.PythonREPL().run,
  )
#*==============================================================================
class ToolsContent:
  
  @tool("Read webpage content")
  def read_content(url: str) -> str:
    """
    Read content from a webpage.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    text_content = soup.get_text()
    return text_content[:5000]

#*==============================================================================

tools_human = agent_tools.tools_human