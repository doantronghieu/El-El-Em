import add_packages
import config
from use_cases import VTC
import toolkit.gradio.chatbot as chatbot_utils
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
                    "xin chào. Tên tôi là Bob.",
                    "bạn có nhớ tên tôi là gì không",

                    "digital marketing là gì",

                    "làm cách nào để đăng ký tài khoản onlinica",
                    "có mấy loại tài khoản onlinica",
                    "các khoá học tại onlinica có thời hạn sử dụng bao lâu",
                    "onlinica có mấy hình thức thanh toán",
                    "có thể thanh toán bằng momo được không",

                    "các khóa học về design",
                    "các khóa học về trí tuệ nhân tạo",
                    "các khóa học về  ai",
                    "các khóa học của nguyễn ngọc tú uyên",
                    "các khóa học của tú uyên",
                    "các khóa học thầy trần anh tuấn dạy",

                    "cách quản lý thời gian",
                    "nguyên lý phối màu",
                ],
                inputs=[chat_app.human_msg],
            )

if __name__ == "__main__":
    # Enable queuing to facilitate streaming intermediate outputs.
    demo.queue()
    demo.launch(share=True)
    # pass

# gradio app.py
