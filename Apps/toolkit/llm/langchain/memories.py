from langchain.memory import (
  ConversationBufferMemory,
)

from langchain_community.chat_message_histories import ChatMessageHistory

import boto3
from boto3.dynamodb.conditions import Key, Attr

class LangChainSessionDynamodbTable:
	def __init__(self, table_name='LangChainSessionTable'):
		self.table_name = table_name
		self.dynamodb = boto3.resource('dynamodb')
		self.table = self.dynamodb.Table(self.table_name)

	def create_table(
   	self,
		key_schema=[
			{"AttributeName": "UserId", "KeyType": "HASH"},
			{"AttributeName": "SessionId", "KeyType": "RANGE"}
		],
		attribute_definitions=[
			{"AttributeName": "UserId", "AttributeType": "S"},
			{"AttributeName": "SessionId", "AttributeType": "S"}
		]
  ):
		"""
		Create a new DynamoDB table with the specified key schema and attribute definitions.
		"""
		self.table.create(
			TableName=self.table_name,
			KeySchema=key_schema,
			AttributeDefinitions=attribute_definitions,
			BillingMode='PAY_PER_REQUEST'
		)
		self.table.meta.client.get_waiter('table_exists').wait(TableName=self.table_name)
	
	def get_user_sessions(self, user_id):
		"""
		Get all sessions for a specific user.
		"""
		response = self.table.query(
			KeyConditionExpression=Key('UserId').eq(user_id)
		)
		return response['Items']

	def get_session_ids(self, user_id):
		"""
		Get all session IDs from a list of chat history items.
		"""
		response = self.table.query(
			KeyConditionExpression=Key('UserId').eq(user_id)
		)
		chat_history = response['Items']
		session_ids = [item['SessionId'] for item in chat_history]
		return session_ids



""" DynamoDB
import boto3

dynamodb = boto3.resource("dynamodb")

table = dynamodb.create_table(
	TableName="LangChainSessionTable",
	KeySchema=[
		{ "AttributeName": "SessionId",	"KeyType": "HASH" },
		{ "AttributeName": "UserId", "KeyType": "RANGE" },
  ],
	AttributeDefinitions=[
		{ "AttributeName": "SessionId", "AttributeType": "S" },
		{ "AttributeName": "UserId", "AttributeType": "S" },
  ],
	BillingMode="PAY_PER_REQUEST",
)

# Wait until the table exists
table.meta.client.get_waiter("table_exists").wait(TableName="LangChainSessionTable")


history = histories.DynamoDBChatMessageHistory(
	table_name="LangChainSessionTable", session_id="0",
	key={
		"SessionId": "0",
		"UserId": "admin",
	}
)

history.add_user_message("Hi!")
history.add_ai_message("Whats up?")

print(history.messages)

# Clear
await history.aclear()
"""

