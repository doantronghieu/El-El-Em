from langchain.memory import (
  ConversationBufferMemory, ChatMessageHistory,
)


"""
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