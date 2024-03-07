import config
from use_cases import VTC
import my_gradio.chatbot as chatbot_utils
import gradio as gr
import warnings
warnings.filterwarnings("ignore")

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

    with gr.Tab("Onlinica"):
        with gr.Blocks() as demo_onlinica:

            chat_app = chatbot_utils.create_chat_app("Onlinica")

            txt_msg = chat_app.human_msg.submit(
                fn=chatbot_utils.user,
                inputs=[chat_app.human_msg, chat_app.chatbot],
                outputs=[chat_app.human_msg, chat_app.chatbot],
                queue=False,
            )
            txt_msg.then(
                fn=chatbot_utils.bot_onlinica,
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

            gr.Examples(
                examples=[
                    "Typography là gì?",
                    "Procreate là gì?",
                    "Cách thiết kế CV ấn tượng với phần mềm Adobe Illustrator",
                    "Làm cách nào để đăng ký tài khoản Onlinica?",
                    "Có mấy loại tài khoản Onlinica?",
                    "Các khoá học của tôi tại Onlinica có thời hạn sử dụng bao lâu?",
                    "Onlinica có mấy hình thức thanh toán?",
                    "Tôi có thể xoá tài khoản Onlinica không?",
                    "Các khóa học về Digital Marketing",
                    "Các khóa học về lập trình",
                ],
                inputs=[chat_app.human_msg],
            )

if __name__ == "__main__":
    # Enable queuing to facilitate streaming intermediate outputs.
    demo.queue()
    demo.launch(share=False)
    # pass

# gradio app.py
