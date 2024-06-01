import add_packages
import yaml, re, os, json, requests
from crewai import Agent, Task
from crewai import Agent, Task
from pprint import pprint
from bs4 import BeautifulSoup
from toolkit.langchain.tools import tool
from toolkit.langchain import (
  document_loaders, text_splitters, documents, tools,
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
    """\
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

  search_duckduckgo = tools.DuckDuckGoSearchRun()
  
#*==============================================================================
class ToolsCalculator():
  
  @tool("Make a calculation")
  def calculate(operation: str):
    """\
    Useful to perform any mathematical calculations, like sum, minus, multiplication, \
    division, etc. 
    
    Parameters:
    - operation: str
    
    The input to this tool should be mathematical expression, examples:
    - `1 + 2`
    - `3 - 1`
    - `1 * 2`
    - `4 / 2`
    - `5 // 2`
    - `6 % 2`
    - `2 ** 3`
    - `(1 + 2) * 3`
    - `(5 - 2) ** 2`
    - `10 // (3 - 1)`
    - `25 % (4 + 1)`
    - `(8 * 2) / (6 - 2)`
    - `(4 ** 3) - (2 * 5)`
    - `(2 * 3) ** (4 % 2)`
    - `(5 + 3) * (12 // 4) - (2 ** 2)`
    - `((7 * 2) - 10) / (3 + (4 // 2))`
    - `(10 - 2) * (16 % 5) + (3 ** 2)`
    - `(2 ** 3) * ((15 // 5) - (4 % 3))`
    - `((6 * 2) + (8 // 4)) ** (7 % 3)`
    - `2 ** (3 + 4)`
    - `4 ** (1 / 2)`
    - `2 ** (3 * 4)`
    
    ALLOWED Action input must be string of mathematical expression, examples: \n
    - "1 + 2"
    
    NOT ALLOWED Action input examples: \n
    - "expression": "1 + 2"
    - "calculation": "7 * 24"
    """
    try:
      return eval(operation)
    except SyntaxError:
      return "Error: Invalid syntax in mathematical expression"
  
  @tool("Python REPL")
  def python_repl(operation: str):
    """\
    A Python shell. Use this to execute python commands. Input should be a 
    valid python command, examples:
    
    - `1 + 2`\n
    - `1 * 2`\n
    - `3 - 1`\n
    - `4 / 2`\n
    - `5 // 2`\n
    - `6 % 2`\n
    - `2 ** 3`\n
    - `(1 + 2) * 3`\n
    - `(5 - 2) ** 2`\n
    - `10 // (3 - 1)`\n
    - `25 % (4 + 1)`\n
    - `(8 * 2) / (6 - 2)`\n
    - `(4 ** 3) - (2 * 5)`\n
    - `(2 * 3) ** (4 % 2)`\n
    - `(5 + 3) * (12 // 4) - (2 ** 2)`\n
    - `((7 * 2) - 10) / (3 + (4 // 2))`\n
    - `(10 - 2) * (16 % 5) + (3 ** 2)`\n
    - `(2 ** 3) * ((15 // 5) - (4 % 3))`\n
    - `((6 * 2) + (8 // 4)) ** (7 % 3)`\n
    - `2 ** (3 + 4)`\n
    - `4 ** (1 / 2)` (square root of 4)\n
    - `pow(2, 3)` (equivalent to `2 ** 3`)\n
    - `abs(-5)` (absolute value of -5)\n
    - `round(3.14159, 2)` (rounding 3.14159 to 2 decimal places)\n
    - `max(5, 9)` (maximum of 5 and 9)\n
    - `min(3, 8)` (minimum of 3 and 8)\n
    - `sum([1, 2, 3, 4, 5])` (sum of a list of numbers)\n
    - `2 ** (3 * 4)` (exponential with multiplication)\n
    - `pow(3, 2)` (equivalent to `3 ** 2`, exponentiation)\n
    - `abs(-10)` (absolute value)\n
    - `round(3.567, 2)` (rounding to 2 decimal places)\n
    - `max(4, 7, 2, 9)` (maximum of a set of numbers)\n
    - `min(8, 2, 6, 4)` (minimum of a set of numbers)\n
    - `sum([1, 2, 3, 4, 5])` (sum of a list of numbers)\n
    """
    my_python_repl = tools.PythonREPL()
    return my_python_repl.run(operation)
  
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

tools_human = tools.tools_human