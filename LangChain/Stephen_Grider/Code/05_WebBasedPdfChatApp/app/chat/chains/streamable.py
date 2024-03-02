from flask import current_app
from queue import Queue
from threading import Thread
from langchain.chains.base import Chain ###
from app.chat.callbacks.stream import StreamingHandler


# class StreamableChain:
class StreamableChain(Chain):
    def stream(self, input):
        queue = Queue()
        streaming_handler = StreamingHandler(queue)
        
        # If not using Flask
        # def task():
        #     self(input, callbacks=[streaming_handler])        
        # Thread(target=task).start()
        ########################################################################
        # If using Flask
        def task(app_context):
            app_context.push()
            self(input, callbacks=[streaming_handler])
        Thread(target=task, args=[current_app.app_context()]).start()
        ########################################################################
        
        while True:
            token = queue.get()
            if token is None:
                break
            yield token
