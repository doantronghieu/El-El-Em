from app.chat.chains.streamable import StreamableChain
from langchain.chains import ConversationalRetrievalChain

class StreamingConversationalRetrievalChain(StreamableChain, 
                                            ConversationalRetrievalChain):
    pass