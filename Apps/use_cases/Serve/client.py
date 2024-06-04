import httpx
import asyncio
import os
import json
from typing import Literal

server_fastapi = os.environ.get('ENDPOINT_PROD_FASTAPI', 'http://127.0.0.1:8000')

async def get_langchain_session_dynamodb_table(
	server_fastapi = server_fastapi, 
	user: str = "admin",
):
	url = f"{server_fastapi}/langchain-session-dynamodb-table"
	headers = {
		"accept": "application/json"
	}
	params = {
		"user": user,
	}

	try:
		response = httpx.get(url, headers=headers, params=params)
		response.raise_for_status()  # This will raise an exception if the request failed (e.g., 404, 500, etc.)
		data = json.loads(response.content.decode("utf-8"))
		return data
	except httpx.HTTPError as e:
		print(f"Request failed with status code {e.response.status_code}: {e.response.content.decode('utf-8')}")
		return None

async def invoke_agent(
	query,
	server_fastapi = server_fastapi,
	history_type: str = "dynamodb",
	user_id=None,
	session_id: str = "default",
):
	url = f"{server_fastapi}/invoke-agent"
	headers = {
		"accept": "application/json"
	}
	params = {
		"query": query,
		"history_type": history_type,
		"user_id": user_id,
		"session_id": session_id,
	}

	try:
		response = httpx.get(url, headers=headers, params=params)
		response.raise_for_status()
		data = json.loads(response.content.decode("utf-8"))
		return data
	except httpx.HTTPError as e:
		print(f"Request failed with status code {e.response.status_code}: {e.response.content.decode('utf-8')}")
		return None

async def stream_agent_async(
	query,
	server_fastapi = server_fastapi,
	history_type: str = "dynamodb",
	user_id=None,
	session_id: str = "default",
):
	url = f"{server_fastapi}/stream-agent"
	params = {
		"query": query,
		"history_type": history_type,
		"user_id": user_id,
		"session_id": session_id,
	}

	async with httpx.AsyncClient() as client:
		try:
			async with client.stream('GET', url, params=params, timeout=60) as r:
				async for chunk in r.aiter_text():
					yield chunk
		except httpx.HTTPError as e:
			print(f"Request failed with status code {e.response.status_code}: {e.response.content.decode('utf-8')}")

def stream_agent_sync(
	query,
	server_fastapi = server_fastapi,
	history_type: str = "dynamodb",
	user_id=None,
	session_id: str = "default",
):
	url = f"{server_fastapi}/stream-agent"
	params = {
		"query": query,
		"history_type": history_type,
		"user_id": user_id,
		"session_id": session_id,
	}

	with httpx.stream('GET', url, params=params, timeout=60) as r:
		for chunk in r.iter_text():
			yield chunk

async def get_chat_history(
	server_fastapi = server_fastapi,
	history_type: str = "dynamodb",
	user_id=None,
	session_id: str = "default",
):
	url = f"{server_fastapi}/agent-chat-history"
	headers = {
		"accept": "application/json"
	}
	params = {
		"history_type": history_type,
		"user_id": user_id,
		"session_id": session_id,
	}

	try:
		response = httpx.get(url, headers=headers, params=params, timeout=60)
		response.raise_for_status()
		data = json.loads(response.content.decode("utf-8"))
		return data
	except httpx.HTTPError as e:
		print(f"Request failed with status code {e.response.status_code}: {e.response.content.decode('utf-8')}")
		return None

async def clear_agent_chat_history(
	server_fastapi = server_fastapi,
	history_type: str = "dynamodb",
	user_id=None,
	session_id: str = "default",
):
	url = f"{server_fastapi}/agent-chat-history"
	headers = {
		"accept": "application/json"
	}
	params = {
		"history_type": history_type,
		"user_id": user_id,
		"session_id": session_id,
	}

	try:
		response = httpx.delete(url, headers=headers, params=params)
		response.raise_for_status()
		data = json.loads(response.content.decode("utf-8"))
		return data
	except httpx.HTTPError as e:
		print(f"Request failed with status code {e.response.status_code}: {e.response.content.decode('utf-8')}")
		return None


def vtc_stream_agent_sync(
	query,
	server_fastapi = server_fastapi,
	history_type: str = "dynamodb",
	user_id=None,
	session_id: str = "default",
):
	url = f"{server_fastapi}/vtc-stream-agent"
	params = {
		"query": query,
		"history_type": history_type,
		"user_id": user_id,
		"session_id": session_id,
	}

	with httpx.stream('GET', url, params=params, timeout=60) as r:
		for chunk in r.iter_text():
			yield chunk
  
def tdtu_stream_agent_sync(
	query,
	server_fastapi = server_fastapi,
	history_type: str = "dynamodb",
	user_id=None,
	session_id: str = "default",
):
	url = f"{server_fastapi}/tdtu-stream-agent"
	params = {
		"query": query,
		"history_type": history_type,
		"user_id": user_id,
		"session_id": session_id,
	}

	with httpx.stream('GET', url, params=params, timeout=60) as r:
		for chunk in r.iter_text():
			yield chunk
...