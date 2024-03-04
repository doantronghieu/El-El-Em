import gradio as gr
import my_gradio.chatbot as chatbot_utils
from use_cases.general_chat import conversation_chain

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
                fn=chatbot_utils.bot,
                inputs=[chat_app.chatbot],
                outputs=[chat_app.chatbot],
                api_name="bot_response",
            ).then(
                lambda: gr.Textbox(interactive=True), None, [chat_app.human_msg], queue=False,
            )

            file_msg = chat_app.upload.upload(
                fn=chatbot_utils.add_file, inputs=[chat_app.chatbot, chat_app.upload], outputs=[
                    chat_app.chatbot], queue=False,
            ).then(
                fn=chatbot_utils.bot, inputs=chat_app.chatbot, outputs=chat_app.chatbot,
            )

            chat_app.clear.click(
                fn=lambda: None, inputs=None, outputs=chat_app.chatbot, queue=False
            )

            chat_app.chatbot.like(chatbot_utils.vote, None, None)

    with gr.Tab("Youtube Transcript Summarizer"):
        chat_app = chatbot_utils.create_chat_app("General Chat")

        """
        - Display the user's message instantly in the chat history while generating
        the chatbot's response.
        - Upon user message submission, events are chained by .then()
        - The input field is re-enabled for sending next messages.
        """
        txt_msg = chat_app.human_msg.submit(
            fn=chatbot_utils.user,
            inputs=[chat_app.human_msg, chat_app.chatbot], outputs=[
                chat_app.human_msg, chat_app.chatbot], queue=False
        ).then(
            fn=chatbot_utils.bot, inputs=[chat_app.chatbot], outputs=[
                chat_app.chatbot], api_name="bot_response",
        ).then(
            lambda: gr.Textbox(interactive=True), None, [chat_app.human_msg], queue=False,
        )

        file_msg = chat_app.upload.upload(
            fn=chatbot_utils.add_file, inputs=[chat_app.chatbot, chat_app.upload], outputs=[
                chat_app.chatbot], queue=False,
        ).then(
            fn=chatbot_utils.bot, inputs=chat_app.chatbot, outputs=chat_app.chatbot,
        )

        chat_app.clear.click(
            fn=lambda: None, inputs=None, outputs=chat_app.chatbot, queue=False
        )

        # Adding thumbs-up/down icons to each bot message enable user (dis)liking
        # of messages by attaching a .like() event.
        chat_app.chatbot.like(chatbot_utils.vote, None, None)

if __name__ == "__main__":
    # Enable queuing to facilitate streaming intermediate outputs.
    demo.queue()
    demo.launch()
    # pass

# gradio main.py
