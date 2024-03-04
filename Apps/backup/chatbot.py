import gradio as gr
import random
import time
import uuid


class ChatbotApp:
    def __init__(self, elem_id=None):
        
        self.elem_id = elem_id or str(uuid.uuid4())
        self.chat_history = []
        # chat_history. stores the entire conversation history
        # [(user_msg, bot_msg), (user_msg, bot_msg), ...]
        self.chatbot = gr.Chatbot(
            [],
            elem_id=self.elem_id,
            bubble_full_width=False,
            avatar_images=("./assets/human.png", "./assets/ai.png"),
            height="80vh",
        )

        with gr.Row():
            # allows user messages and triggers chatbot response.
            self.human_msg = gr.Textbox(
                scale=4,
                show_label=False,
                placeholder="Enter your query",
                container=False,
            )

            self.upload = gr.UploadButton(
                "📁", file_types=["image", "video", "audio"])
            
            # clears Textbox and Chatbot history.
            self.clear = gr.ClearButton([self.human_msg, self.chatbot])


def vote(data: gr.LikeData):
    """
    handles user clicks on icons
    argument contains information about the liked/disliked message
    """
    if data.liked:
        print(f"👍 {data.value}, {data.index}, {data.liked}")
    else:
        print(f"👎 {data.value}, {data.index}, {data.liked}")


def add_file(chat_history, file):
    chat_history = chat_history + [((file.name,), None)]
    return chat_history


def user(human_msg, chat_history):
    """
    clear msg box, return new chat_history to bot
    updates the chatbot, clears the input, and renders the field 
    non-interactive to prevent additional messages while the chatbot responds. 
    Instant execution is achieved by setting queue=False, bypassing any 
    potential queue. The chatbot's history appends (user_message, None), 
    indicating an unanswered bot response.
    """
    chat_history = chat_history + [[human_msg, None]]
    return gr.Textbox(value="", interactive=False), chat_history


def bot(chat_history):
    # print(chat_history)
    # [['hello', 'How are you?'], ['hi', None]]
    """
    updates the chatbot's history. The None message is replaced with 
    the bot's response character by character. Gradio automatically converts any
    function with the yield keyword into a streaming output interface.
    """
    ai_msg = random.choice(["How are you?", "I love you", "I'm very hungry"])
    chat_history[-1][1] = ""

    for character in ai_msg:
        chat_history[-1][1] += character
        time.sleep(0.05)
        yield chat_history

# Function to create chat app
def create_chat_app(elem_id=None):
    return ChatbotApp(elem_id)
