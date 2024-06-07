import json
from typing import Union
import asyncio
import os

import add_packages

import streamlit as st
from streamlit_feedback import streamlit_feedback

from toolkit.langchain import (
  prompts, smiths
)
from use_cases.Serve import client
from toolkit.streamlit import utils
from toolkit.streamlit.utils import CHAT_ROLE

#*==============================================================================

st.set_page_config(
  layout="wide",
  page_title="VTC AI"
)

current_file_path = os.path.abspath(__file__)
parent_path = os.path.dirname(current_file_path)
parent_dir = parent_path.split("/")[-1]

if parent_dir != "pages":
  st.sidebar.page_link(f"main.py", label="Home")
  
  if os.getenv("STREAMLIT_GENERAL_CHAT"):
    st.sidebar.page_link(f"pages/01_general_chat.py", label="General Chat")
  if os.getenv("STREAMLIT_DATA_DO_ANYTHING"):
    st.sidebar.page_link(f"pages/02_Data_do_anything.py", label="Do Anything w/ Data")
  if os.getenv("STREAMLIT_GENERATE_ANYTHING"):
    st.sidebar.page_link(f"pages/03_Generate_anything.py", label="Generate Anything")
  if os.getenv("STREAMLIT_VTC"):
    st.sidebar.page_link(f"pages/VTC.py", label="VTC")
  
  st.sidebar.divider()

st.sidebar.image(f"{add_packages.APP_PATH}/assets/logo-vtc.png")
#*==============================================================================

STATES = {
  "USER_EMAIL": {
    "INITIAL_VALUE": None,
  },
  "USER_NAME": {
    "INITIAL_VALUE": None,
  },
  "MESSAGES": {
    "INITIAL_VALUE": [],
  },
  "CONTAINER_PLACEHOLDER": {
    "INITIAL_VALUE": None,
  },
  "LAST_RUN": {
    "INITIAL_VALUE": None,
  },
  "PROMPT_EXAMPLE": {
    "INITIAL_VALUE": None,
  },
  "SELECTED_CHAT": {
    "INITIAL_VALUE": None,
  },
  "BTN_NEW_CHAT": {
    "INITIAL_VALUE": "widget",
  },
  "BTN_CLEAR_CHAT_HISTORY": {
    "INITIAL_VALUE": "widget",
  },
  "BTN_LOGOUT": {
    "INITIAL_VALUE": "widget",
  },
}

utils.initialize_session_state(STATES)

#*==============================================================================

PROJECT_LS = "default" # LangSmith
ENDPOINT_LC = "https://api.smith.langchain.com" # LangChain
CLIENT_LC = smiths.Client(
  api_url=ENDPOINT_LC, api_key=os.getenv("LANGCHAIN_API_KEY")
)
TRACER_LS = smiths.LangChainTracer(project_name=PROJECT_LS, client=CLIENT_LC)
RUN_COLLECTOR = smiths.RunCollectorCallbackHandler()

#*==============================================================================

@st.cache_data(show_spinner=False)
def get_LC_run_url(run_id):
  try:
    result = CLIENT_LC.read_run(run_id).url
  except:
    result = None
    
  return result

def create_callbacks() -> list:
  st_callback = utils.StreamlitCallbackHandler(st.container())
  callbacks = [st_callback]
  return callbacks

#*==============================================================================

async def process_on_user_input(
  prompt: str, 
):
  # Clear the container before displaying user's message
  if st.session_state.container_placeholder is not None:
    st.session_state.container_placeholder.empty()
  
  st.chat_message(CHAT_ROLE.user).markdown(prompt)
  stream = client.vtc_stream_agent_sync(
    query=prompt,
    user_id=st.session_state[STATES["USER_EMAIL"]["KEY"]],
  )
  st.chat_message(CHAT_ROLE.assistant).write_stream(stream)
  
  st.rerun()
  
async def render_chat_messages_on_rerun():
  chat_history = await client.get_chat_history(
    user_id=st.session_state[STATES["USER_EMAIL"]["KEY"]],
  )
  for msg in chat_history:
    msg: Union[prompts.AIMessage, prompts.HumanMessage]
    if "[TOOL - RESULT]" in msg["content"]:
      with st.expander("Observation"):
        part_ignore = "`[TOOL - CALLING]`"
        tool_output = msg["content"][len(part_ignore):]
        try:
          tool_output_json = json.loads(tool_output)
          st.write(tool_output_json)
        except:
          st.write(tool_output)
          
    else:
      st.chat_message(msg["type"]).markdown(msg["content"])

async def on_click_btn_clear_chat_history(
  
):
  await client.clear_agent_chat_history(
    user_id=st.session_state[STATES["USER_EMAIL"]["KEY"]],
  )
  del st.session_state[STATES["LAST_RUN"]["KEY"]]
  st.toast(":orange[History cleared]", icon="🗑️")
#*==============================================================================

if st.session_state[STATES["USER_EMAIL"]["KEY"]]:
  asyncio.run(render_chat_messages_on_rerun())

with st.sidebar:
  if st.session_state[STATES["USER_NAME"]["KEY"]]:
    st.sidebar.write(f'Welcome, :green[{st.session_state[STATES["USER_NAME"]["KEY"]]}]!')
    
    if st.button("Logout"):
      st.session_state[STATES["USER_EMAIL"]["KEY"]] = None
      st.session_state[STATES["USER_NAME"]["KEY"]] = None
      st.session_state["Logout"] = False
      st.rerun()
  else:
    email = st.sidebar.text_input("Email")
    username = email.split("@")[0]
    
    if st.sidebar.button("Login"):
      st.session_state[STATES["USER_EMAIL"]["KEY"]] = email
      st.session_state[STATES["USER_NAME"]["KEY"]] = username
      
      st.rerun()
  
  if st.session_state[STATES["USER_EMAIL"]["KEY"]]:
    prompt_example = st.selectbox(
      label="Examples",
      label_visibility="collapsed",
      help="Example prompts",
      placeholder="Examples",
      index=None,
      options=[
        # VTCA FAQs
        "🟢 VTCA FAQs 🟢",
        "Các đặc quyền của học viên VTC Academy",
        "Thành tựu của VTC Academy",
        "Sứ mệnh của VTC Academy",
        "Tầm nhìn của VTC Academy",
        "Mục tiêu của VTC Academy",
        "Thông tin liên hệ các cơ sở của VTC Academy",
        "Mô hình đào tạo tại VTC Academy",
        "Không gian tại VTC Academy",
        "Đội ngũ giảng viên tại VTC Academy",
        "Đối tượng tuyển sinh tại VTC Academy",
        "Hình thức tuyển sinh tại VTC Academy",
        "Quy trình đăng ký và làm hồ sơ nhập học tại VTC Academy",
        "Hồ sơ nhập học trực tuyến tại VTC Academy",
        "Lịch khai giảng các khoá học tại VTC Academy",
        "Tôi muốn nộp hồ sơ nhập học online cho con tôi vì đang ở một thành phố khác, làm sao có thể làm thủ tục nhập học tại VTC Academy?",
        "Hồ sơ nhập học trực tuyến (online) tại VTC Academy cần có những gì?",
        "Làm sao để biết con tôi đã được nhập học thành công tại VTC Academy?",
        "Lịch khai giảng của Nhà trường tại VTC Academy là bao giờ?",
        "Nếu học online thì chất lượng và lộ trình học tập có đảm bảo không? Hình thức học online tại VTC Academy như thế nào?",
        "Học viên có cần phải thi tuyển đầu vào tại VTC Academy không?",
        "Tôi muốn tham quan cơ sở nhưng vì ở xa nên không có điều kiện đến tận trường. Có thể chia sẻ hình ảnh môi trường học tập thực tế tại các cơ sở ",
        "VTC Academy có các hình thức hoặc chương trình ưu đãi nào để hỗ trợ gia đình học viên không?",
        "Các hình thức đóng học phí cho nhà trường tại VTC Academy?",
        "Tôi muốn đóng học phí trực tiếp cho trường VTC Aacademy vì không đủ điều kiện để đóng qua Internet Banking hay qua thẻ thì nhà trường có hỗ ",
        "Nhà trường tại VTC Academy có hỗ trợ tìm nhà trọ không?",
        "Sau khi hoàn thành các khoá học tại VTC Academy, học viên có việc làm ngay không?",
        "Đăng ký xét học bạ tại VTC Academy",
        "VTC Academy Plus tại VTC Academy",
        "Học bổng tài năng tại VTC Academy",
        "Học bổng tài năng tại VTC Academy Plus",
        "Thông tin tham khảo về tuyển sinh ở VTC Academy",
        "Hình thức xét học bạ tại VTC Academy như thế nào?",
        "Điều kiện đăng ký xét tuyển các ngành học tại VTC Academy là gì?",
        "Tôi không có bằng THPT, liệu có thể học tại VTC Academy được không?",
        "VTC Academy có hỗ trợ nơi ở cho những bạn ở xa theo học tại trường không?",
        "Để nhập học tại VTC Academy cần thi khối nào, môn thi là gì, điểm chuẩn là bao nhiêu?",
        "Sau khi xét học bạ tại VTC Academy thì có cần đăng ký trên cổng thông tin bộ giáo dục không?",
        "Bằng tốt nghiệp tại VTC Academy thuộc hệ gì?",
        "Bằng tốt nghiệp tại VTC Academy có xin được bên ngoài không hay chỉ doanh nghiệp liên kết với trường?",
        "Khi nào thì nhận được chứng chỉ DMI tại VTC Academy?",
        "Bằng khóa Plus tại VTC Academy khác gì so với khóa thường?",
        "Trường tại VTC Academy có các cơ sở nào?",
        "Giấy tờ đăng ký và nhập học tại VTC Academy cần những gì?",
        "Thời gian học tại VTC Academy có phù hợp cho sinh viên học 2 trường/người đi làm không?",
        "Sau khi tốt nghiệp, tôi có thể làm việc ở những công ty nào mà VTC Academy liên kết?",
        "Học phí giữa các cơ sở tại VTC Academy có chênh lệch không?",
        "Thời gian học 1 khóa học tại VTC Academy khoảng mấy năm?",
        "Thời gian nhập học tại VTC Academy là khi nào?",
        "Hạn chót gửi hồ sơ xét tuyển tại VTC Academy là khi nào?",
        "Tôi có thể dời thời gian nhập học tại VTC Academy không? Nếu có thì được dời lại trong bao lâu?",
        "Khi đăng ký nhập học tại VTC Academy có được miễn nghĩa vụ quân sự không?",
        "Trường tại VTC Academy có lớp buổi tối không?",
        "Em không biết chọn ngành nào tại VTC Academy?/Em không biết mình phù hợp với ngành nào?",
        "Tôi chưa đủ tuổi thì có được theo học tại VTC Academy không?",
        "VTC Academy có khóa học online không?",
        "Tôi học song song tại VTC Academy với trường trung học có được không? Yêu cầu gì?",
        "VTC Academy có cam kết việc làm bằng văn bản pháp lý thật không?",
        "Học tại VTC Academy có liên thông được không?",
        "Trong thời gian học hệ Plus tại VTC Academy có đi học ở các trường liên kết nước ngoài không?",
        "Trường có tăng học phí trong thời gian học tại VTC Academy không?",
        "Trường đang có các ưu đãi gì khi đăng ký nhập học tại VTC Academy không?",
        "Thông tin chi tiết về trả góp học phí 0% tại VTC Academy ",
        "Mê game thì có thể học những ngành học nào tại VTC Academy?",
        "Có chương trình hỗ trợ học phí tại VTC Academy hay ưu đãi gì không?",
        "Có ưu đãi gì cho Quân Nhân Xuất Ngũ tại VTC Academy không?",
      
        # VTCA courses
        "🟢 VTCA courses 🟢",
        "Các khối ngành đào tạo tại VTCA?",
        "Khối ngành thiết kế tại VTC Academy có những chuyên ngành nào?",
        "Thời lượng và hình thức học của khóa học lập trình phần mềm tại VTCA",
        "Ở VTCA có những khoá học dài hạn nào?",
        "Ở VTCA có những khoá học trung hạn nào?",
        "Đối tượng nhập học của khóa học thiết kế 3D tại VTC Academy",
        "Các khoá học tại VTCA có hình thức học tập trung",
        "Link tham khảo khóa học quản lý chuỗi ứng dụng và logistic tại VTC Academy",
        "Học phí cho khóa học Digital Marketing VTCA",
        "Thời gian khai giảng khóa học 3D Modeling tại VTCA",
        "Cơ hội nghề nghiệp của ngành học digital marketing ở VTCA?",
        "Các chương trình học tiêu chuẩn tại VTCA",
        "Các chương trình học plus của VTCA",
        
        # Onlinica courses
        "🟢 Onlinica courses 🟢",
        "Các thể loại khoá học ở Onlinica", # redo onli_faq
        "Giảng viên nào có nhiều khoá học nhất ở Onlinica",
        "Các Giảng viên có nhiều khoá học nhất ở Onlinica",
        "Thể loại khoá học nào có nhiều khoá học nhất ở Onlinica?",
        "Các khoá học của thầy Trần Anh Tuấn ở Onlinica", 
        "Các khoá học về lập trình ở Onlninica",
        "Các giảng viên dạy về lập trình ở Onlninica",
        "Thông tin khoá học kỹ năng quản lý thời gian ở Onlinica?",
        "Giảng viên của khoá học 'Kỹ Năng Giao Tiếp Hiệu Quả' ở Onlinica",
        "Các khoá học về game ở Onlinica", #
      
        # Onlinica FAQs
        "🟢 Onlinica FAQs 🟢",
        "Làm cách nào để đăng ký tài khoản Onlinica?", 
        "Có thể thanh toán Onlinica bằng Momo được không",
        "Có mấy loại tài khoản Onlinica?",
        "Các khoá học ở Onlinica có mất phí không?",
        "Các khoá học của tôi tại Onlinica có thời hạn sử dụng bao lâu?",
        "Tôi có thể xoá tài khoản Onlinica không? Và làm như thế nào?",
        "Tôi có bị mất phí khi hết thời gian trải nghiệm Onlinica VIP không?",
        "Tôi sẽ nhận được chứng chỉ hoàn thành khóa học chứ? Và sẽ nhận qua hình thức nào?",
        "Vì sao tôi chưa nhận được chứng chỉ hoàn thành khi đã học xong?",
        "Onlinica có mấy hình thức thanh toán?",
        "Làm cách nào để đăng ký tài khoản Onlinica?",
        "Có mấy loại tài khoản Onlinica?",
        "Các khoá học ở Onlinica có mất phí không?",
        "Các khoá học của tôi tại Onlinica có thời hạn sử dụng bao lâu?",
        "Tôi có thể xoá tài khoản Onlinica không? Và làm như thế nào?",
        "Tôi có bị mất phí khi hết thời gian trải nghiệm Onlinica VIP không?",
        "Onlinica hiện đang có các khóa học nào?",
        "Tôi sẽ nhận được chứng chỉ hoàn thành khóa học chứ? Và sẽ nhận qua hình thức nào?",
        "Vì sao tôi chưa nhận được chứng chỉ hoàn thành khi đã học xong?",
        "Onlinica có mấy hình thức thanh toán?",
      ],
      key=STATES["PROMPT_EXAMPLE"]["KEY"],
    )  
    
    btn_clear_chat_history = st.button(
      label="🗑️", 
      help="Clear Chat History",
      key=STATES["BTN_CLEAR_CHAT_HISTORY"]["KEY"],
    )
    if btn_clear_chat_history:
      asyncio.run(on_click_btn_clear_chat_history())
      st.rerun()
  
#*----------------------------------------------------------------------------

prompt: Union[str, None] = st.chat_input(
  "Say something",
  disabled=st.session_state[STATES["USER_EMAIL"]["KEY"]] is None,
)

if st.session_state[STATES["USER_EMAIL"]["KEY"]] and prompt_example:
  prompt = prompt_example
  del st.session_state[STATES["PROMPT_EXAMPLE"]["KEY"]]
  
if prompt:
  asyncio.run(process_on_user_input(prompt))

# st.write(st.session_state)

# streamlit run VTC.py