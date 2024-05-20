import httpx
import asyncio
import json
from typing import Literal

server_fastapi = 'http://127.0.0.1:8000'

async def invoke_agent(
	query,
	server_fastapi = 'http://127.0.0.1:8000', 
):
	"""
	await invoke_agent("Tell me a long joke")
 	"""
  
	url = f"{server_fastapi}/invoke-agent"
	headers = {
		"accept": "application/json"
	}
	params = {
		"query": query
	}

	try:
		response = httpx.get(url, headers=headers, params=params)
		response.raise_for_status()  
		res_txt = response.text
		return res_txt
	except httpx.HTTPError as e:
		print(f"Request failed with status code {e.response.status_code}: {e.response.content.decode('utf-8')}")
		return None

async def stream_agent_async(
  query,
	server_fastapi = 'http://127.0.0.1:8000', 
):
	url = f"{server_fastapi}/stream-agent"
	params = {
		"query": query,
	}

	async with httpx.AsyncClient() as client:
		try:
			async with client.stream('GET', url, params=params, timeout=60) as r:
				async for chunk in r.aiter_text():  # or, async for line in r.aiter_lines():
					yield chunk
		except httpx.HTTPError as e:
			print(f"Request failed with status code {e.response.status_code}: {e.response.content.decode('utf-8')}")

def stream_agent_sync(
  query,
	server_fastapi = 'http://127.0.0.1:8000', 
):
	url = f"{server_fastapi}/stream-agent"
	params = {
		"query": query,
	}

	with httpx.stream('GET', url, params=params, timeout=60) as r:
		for chunk in r.iter_text():  # or, for line in r.iter_lines():
				yield chunk

async def get_chat_history(
  server_fastapi = 'http://127.0.0.1:8000', 
):
	url = f"{server_fastapi}/agent-chat-history"
	headers = {
		"accept": "application/json"
	}
	params = {
	}

	try:
		response = httpx.get(url, headers=headers, params=params)
		response.raise_for_status()  # This will raise an exception if the request failed (e.g., 404, 500, etc.)
		data = json.loads(response.content.decode("utf-8"))
		return data
	except httpx.HTTPError as e:
		print(f"Request failed with status code {e.response.status_code}: {e.response.content.decode('utf-8')}")
		return None

...