from operator import contains
from turtle import ht
from bs4 import BeautifulSoup
import requests
import sqlite3
from pydantic import BaseModel  # from pydantic.v1 import BaseModel
from typing import List
from langchain.tools import Tool
import urllib.request


class ExtractWebsiteArgsSchema(BaseModel):
    url: str

def fetch_content(url):
    """
    Fetch content from the given URL using either requests or urllib.
    """
    try:
        if 'html' in url:
            response = requests.get(url)
            response.raise_for_status()
            return response.content
        else:
            with urllib.request.urlopen(url) as fp:
                return fp.read()
    except (requests.RequestException, urllib.error.URLError) as e:
        return f"An error occurred: {e}"


def process_html_content(html_content):
    """
    Process HTML content using BeautifulSoup and return the text content.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    return soup.get_text(separator='\n', strip=True)


def write_content_to_file(filename, content):
    """
    Write the given content to a file.
    """
    file_path = f'{filename}.txt'
    file_path = 'company_info.txt'
    
    with open(file_path, 'w', encoding="utf-8") as f:
        f.write(content)
    print(f'Five saved in: "{file_path}"')

def extract_website_content(url):
    """
    Extract content from a website and write it to a file.
    """
    website_name = url.split('//')[-1].split('/')[0].replace('.', '-')
    content = fetch_content(url)

    if isinstance(content, str) and "An error occurred" in content:
        return content

    text_content = process_html_content(content)
    write_content_to_file(website_name, text_content)
    return text_content

# Example usage
content = extract_website_content('https://vtc.edu.vn/giang-vien/')
content = extract_website_content(
    'https://www.presight.io/privacy-policy.html')


extract_website_content_tool = Tool.from_function(
    name='extract_website_content',
    description='Given a url of a website, return all the text from the website',
    func=extract_website_content,
    args_schema=ExtractWebsiteArgsSchema
)
