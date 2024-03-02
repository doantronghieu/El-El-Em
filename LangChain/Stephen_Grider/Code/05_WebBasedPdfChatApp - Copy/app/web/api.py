from typing import Dict
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from app.web.db import db
from app.web.db.models import Message
from app.web.db.models.conversation import Conversation


def get_messages_by_conversation_id(
    conversation_id: str,
) -> AIMessage | HumanMessage | SystemMessage:
    """
    Finds all messages that belong to the given conversation_id

    :param conversation_id: The id of the conversation

    :return: A list of messages
    """
    messages = (
        db.session.query(Message)
        .filter_by(conversation_id=conversation_id)
        .order_by(Message.created_on.desc())
    )
    return [message.as_lc_message() for message in messages]


def add_message_to_conversation(
    conversation_id: str, role: str, content: str
) -> Message:
    """
    Creates and stores a new message tied to the given conversation_id
        with the provided role and content

    :param conversation_id: The id of the conversation
    :param role: The role of the message
    :param content: The content of the message

    :return: The created message
    """
    return Message.create(
        conversation_id=conversation_id,
        role=role,
        content=content,
    )


def get_conversation_components(conversation_id: str) -> Dict[str, str]:
    """
    Returns the components used in a conversation
    """
    conversation = Conversation.find_by(id=conversation_id)
    return {
        "llm": conversation.llm,
        "retriever": conversation.retriever,
        "memory": conversation.memory,
    }


def set_conversation_components(
    conversation_id: str, llm: str, retriever: str, memory: str
) -> None:
    """
    Sets the components used by a conversation
    """
    conversation = Conversation.find_by(id=conversation_id)
    conversation.update(llm=llm, retriever=retriever, memory=memory)
