import add_packages
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

from my_streamlit import utils

#*==============================================================================

st.set_page_config(
  layout="wide",
)

with open(f"{add_packages.APP_PATH}/app_streamlit/.streamlit/users.yaml", "r+") as f:
  config = yaml.load(f, Loader=SafeLoader)

STATES = {
  "MESSAGES": {
    "INITIAL_VALUE": [],
  },
  "AUTHENTICATION_STATUS": {
    "INITIAL_VALUE": "widget",
  },
  "BTN_REGISTER": {
    "INITIAL_VALUE": "widget",
  },
  "IS_BTN_REGISTER": {
    "INITIAL_VALUE": False,
  },
  "BTN_FORGOT_USERNAME": {
    "INITIAL_VALUE": "widget",
  },
  "IS_BTN_FORGOT_USERNAME": {
    "INITIAL_VALUE": False,
  },
  "BTN_FORGOT_PASSWORD": {
    "INITIAL_VALUE": "widget",
  },
  "IS_BTN_FORGOT_PASSWORD": {
    "INITIAL_VALUE": False,
  },
  "BTN_RESET_PASSWORD": {
    "INITIAL_VALUE": "widget",
  },
  "IS_BTN_RESET_PASSWORD": {
    "INITIAL_VALUE": False,
  },
  "BTN_UPDATE_USER_DETAIL": {
    "INITIAL_VALUE": "widget",
  },
  "IS_BTN_UPDATE_USER_DETAIL": {
    "INITIAL_VALUE": False,
  },
}

utils.initialize_session_state(STATES)

#*==============================================================================

st.title("Welcome to My Home")

authenticator = stauth.Authenticate(
  config['credentials'],
  config['cookie']['name'],
  config['cookie']['key'],
  config['cookie']['expiry_days'],
  config['pre-authorized']
)

def save_config():
  # Ensure that the configuration file is re-saved anytime the credentials are 
  # updated or whenever the reset_password, register_user, forgot_password, or 
  # update_user_details widgets are used.
  with open(f"{add_packages.APP_PATH}/app_streamlit/.streamlit/users.yaml", "w") as f:
    
    yaml.dump(config, f, default_flow_style=False)
    

def handle_toggle_btn(btn_name: str):
  st.session_state[f"is_{btn_name}"] = not st.session_state[f"is_{btn_name}"]

def handle_on_click_btn_register():
  try:
    reg_user_email, reg_user_username, \
      reg_user_name = authenticator.register_user(
        pre_authorization=False,
        location="sidebar",
        domains=['gmail.com'],
      )
    if reg_user_email:
      st.sidebar.success('User registered successfully')
      handle_toggle_btn(STATES["BTN_REGISTER"]["KEY"])
      save_config()
    
      
  except Exception as e:
    print(e)
    st.error(e)
  
def handle_on_click_btn_forgot_username():
  try:
    forgot_username_username, forgot_username_email = authenticator.forgot_username(
      location="sidebar",
    )
    if forgot_username_username:
      st.success('Username to be sent securely')
      # The developer should securely transfer the username to the user.
    elif forgot_username_username == False:
      st.error('Email not found')
  except Exception as e:
    st.error(e)

def handle_on_click_btn_forgot_password():
  try:
    forgot_pw_username, forgot_pw_email, \
      new_random_password = authenticator.forgot_password(
        location="sidebar"
      )
    if forgot_pw_username:
      st.success('New password to be sent securely')
      # The developer should securely transfer the new password to the user.
    elif forgot_pw_username == False:
      st.error('Username not found')
  except Exception as e:
    st.error(e)

def handle_on_click_btn_reset_password():
  try:
    if authenticator.reset_password(
      username=st.session_state["username"],
      location="sidebar",
    ):
      st.success('Password modified successfully')
  except Exception as e:
    st.error(e)

def handle_on_click_btn_update_user_detail():
  try:
    if authenticator.update_user_details(
      username=st.session_state["username"],
      location="sidebar",
    ):
      st.success('Entries updated successfully')
  except Exception as e:
    st.error(e)
#*==============================================================================

# re-invoked on every page
user_name, user_authen_status, user_username = authenticator.login(
  location="sidebar", # main, sidebar
)

if st.session_state["authentication_status"]:
  st.sidebar.write(f'Welcome *{st.session_state["name"]}*')
  
  authenticator.logout(key="logout_btn", location="sidebar")
  
  st.title('Some content')
  
elif st.session_state["authentication_status"] == False:
  st.sidebar.error('Username/password is incorrect')
  
elif st.session_state["authentication_status"] is None:
  st.sidebar.warning('Please enter your username and password')

  btn_register = st.sidebar.button(
    "Register", key=STATES["BTN_REGISTER"]["KEY"],
    on_click=handle_toggle_btn, 
    kwargs=dict(btn_name=STATES["BTN_REGISTER"]["KEY"]),
  )
  if st.session_state[STATES["IS_BTN_REGISTER"]["KEY"]]:
    handle_on_click_btn_register()
  
  btn_forgot_username = st.sidebar.button(
    "Forgot Username", key=STATES["BTN_FORGOT_USERNAME"]["KEY"],
    on_click=handle_toggle_btn, 
    kwargs=dict(btn_name=STATES["BTN_FORGOT_USERNAME"]["KEY"]),
  )
  if st.session_state[STATES["IS_BTN_FORGOT_USERNAME"]["KEY"]]:
    handle_on_click_btn_forgot_username()
    
  btn_forgot_password = st.sidebar.button(
    "Forgot Password", key=STATES["BTN_FORGOT_PASSWORD"]["KEY"],
    on_click=handle_toggle_btn, 
    kwargs=dict(btn_name=STATES["BTN_FORGOT_PASSWORD"]["KEY"]),
  )
  if st.session_state[STATES["IS_BTN_FORGOT_PASSWORD"]["KEY"]]:
    handle_on_click_btn_forgot_password()
    
if st.session_state["authentication_status"]:
  btn_reset_password = st.sidebar.button(
    "Reset Password", key=STATES["BTN_RESET_PASSWORD"]["KEY"],
    on_click=handle_toggle_btn,
    kwargs=dict(btn_name=STATES["BTN_RESET_PASSWORD"]["KEY"]),
  )
  if st.session_state[STATES["IS_BTN_RESET_PASSWORD"]["KEY"]]:
    handle_on_click_btn_reset_password()
  
  btn_update_user_detail = st.sidebar.button(
    "Update User Detail", key=STATES["BTN_UPDATE_USER_DETAIL"]["KEY"],
    on_click=handle_toggle_btn, 
    kwargs=dict(btn_name=STATES["BTN_UPDATE_USER_DETAIL"]["KEY"]),
  )
  if st.session_state[STATES["IS_BTN_UPDATE_USER_DETAIL"]["KEY"]]:
    handle_on_click_btn_update_user_detail()
  


st.write(st.session_state)
st.write(config)