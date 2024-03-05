import warnings
warnings.filterwarnings("ignore")

import gradio as gr
import uuid
import my_gradio.chatbot as chatbot_utils
from use_cases import general_chat


with gr.Blocks() as demo:
    with gr.Tab("General Chat"):
        with gr.Blocks() as demo_general_chat:

            chat_app = chatbot_utils.create_chat_app("General Chat")

            txt_msg = chat_app.human_msg.submit(
                fn=chatbot_utils.user,
                inputs=[chat_app.human_msg, chat_app.chatbot],
                outputs=[chat_app.human_msg, chat_app.chatbot],
                queue=False,
            )
            txt_msg.then(
                fn=chatbot_utils.bot_general_chat,
                inputs=[chat_app.chatbot, chat_app.human_msg],
                outputs=[chat_app.chatbot],
                api_name="bot_response",
            )

            txt_msg.then(
                lambda: gr.Textbox(interactive=True), None, [chat_app.human_msg], queue=False,
            )
            txt_msg.then(
                fn=chatbot_utils.clean_human_msg,
                inputs=[chat_app.human_msg],
                outputs=[chat_app.human_msg],
            )

            file_msg = chat_app.upload.upload(
                fn=chatbot_utils.add_file, inputs=[chat_app.chatbot, chat_app.upload], outputs=[
                    chat_app.chatbot], queue=False,
            ).then(
                fn=chatbot_utils.bot_general_chat, inputs=chat_app.chatbot, outputs=chat_app.chatbot,
            )

            chat_app.clear.click(
                fn=lambda: None, inputs=None, outputs=None, queue=False
            )

            chat_app.chatbot.like(chatbot_utils.vote, None, None)

    with gr.Tab("Youtube Transcript Summarizer"):
        pass

if __name__ == "__main__":
    # Enable queuing to facilitate streaming intermediate outputs.
    demo.queue()
    demo.launch(share=True)
    # pass

# gradio app.py
