import secrets
import string
import time

import streamlit as st
import yaml
from hashlib import sha256
from secrets import token_hex

st.set_page_config(layout="wide")

class UserUtils:
    HASH_ALGORITHM = "sha256"
    SALT_LENGTH = 16

    @staticmethod
    def hash_password(password: str, salt: str = None) -> str:
        if salt is None:
            salt = token_hex(UserUtils.SALT_LENGTH)
        if UserUtils.HASH_ALGORITHM == "sha256":
            return f"{UserUtils.HASH_ALGORITHM}${UserUtils.SALT_LENGTH}${salt}${sha256((salt + password).encode()).hexdigest()}"
        raise ValueError(f"Unsupported hash algorithm: {UserUtils.HASH_ALGORITHM}")

    @staticmethod
    def generate_password(length=12, complexity=SALT_LENGTH):
        char_sets = [
            string.ascii_lowercase,
            string.ascii_letters,
            string.ascii_letters + string.digits,
            string.ascii_letters + string.digits + string.punctuation,
        ]
        complexity = min(max(complexity, 1), len(char_sets))
        char_set = char_sets[complexity - 1]
        return ''.join(secrets.choice(char_set) for _ in range(length))

class UserManager:
    def __init__(self, config_file="users.yaml"):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self):
        with open(self.config_file, "r") as f:
            config = yaml.safe_load(f)
        return config.get("auth", {}).get("users", {})

    def save_config(self, config):
        with open(self.config_file, "w") as f:
            yaml.dump({"auth": {"users": config}}, f, default_flow_style=False)

    def authenticate_user(self, username: str, password: str) -> bool:
        user = self.config.get(username)
        if user is None:
            return False
        hashed_password, stored_salt = user["password_hash"].split("$")[3], user["password_hash"].split("$")[2]
        if UserUtils.hash_password(password, stored_salt) == user["password_hash"]:
            st.session_state.user = user
            return True
        user["failed_login_attempts"] += 1
        self.config[username] = user
        return False

    def register_user(self, email: str, name: str, password: str):
        username = email.split("@")[0]
        if username in self.config:
            st.sidebar.error("Username already exists!")
            return
        password_hash = UserUtils.hash_password(password)
        self.config[username] = {
            "name": name,
            "email": email,
            "password_hash": password_hash,
            "failed_login_attempts": 0,
            "locked": False,
        }
        st.sidebar.success(f"User '{username}' registered successfully!")
        time.sleep(1)
        st.session_state.email = ""
        st.session_state.name = ""
        st.session_state.password = ""
        self.save_config(self.config)
        st.session_state.registration_success = True
        st.rerun()

    def forgot_password(self, username: str):
        if username not in self.config:
            st.sidebar.error("Username not found!")
            return
        new_password = UserUtils.generate_password()
        password_hash = UserUtils.hash_password(new_password)
        self.config[username]["password_hash"] = password_hash
        st.sidebar.success(f"New password for '{username}' is '{new_password}'")
        self.save_config(self.config)

class App:
    def __init__(self):
        self.manager = UserManager()
        self.init_session_state()

    def init_session_state(self):
        if "user" not in st.session_state:
            st.session_state.user = None
        if "unauth_selected_menu_choice" not in st.session_state:
            st.session_state.unauth_selected_menu_choice = None
        if "auth_selected_menu_choice" not in st.session_state:
            st.session_state.auth_selected_menu_choice = None
        if "email" not in st.session_state:
            st.session_state.email = ""
        if "name" not in st.session_state:
            st.session_state.name = ""
        if "password" not in st.session_state:
            st.session_state.password = ""
        if "registration_success" not in st.session_state:
            st.session_state.registration_success = False

    def render_sidebar(self):
        if st.session_state.user:
            self.render_authenticated_sidebar()
        else:
            self.render_unauthenticated_sidebar()

    def render_authenticated_sidebar(self):
        # st.sidebar.button("Logout", on_click=self.logout, key="logout_button")
        
        auth_menu = ["User Menu", "Logout", "Change Password"]
        st.session_state.auth_selected_menu_choice = st.sidebar.selectbox(
          label="Auth Menu", 
          label_visibility="collapsed",
          options=auth_menu, 
          placeholder="Auth Options",
        )
        
        if st.session_state.auth_selected_menu_choice == "Logout":
          st.sidebar.button("Confirm Logout", on_click=self.logout, key="logout_button")
        elif st.session_state.auth_selected_menu_choice == "Change Password":
          self.render_change_password_form()

    def render_unauthenticated_sidebar(self):
        unauth_menu = ["Login", "Register", "Forgot Password"]
        st.session_state.unauth_selected_menu_choice = st.sidebar.selectbox(
          label="UnAuth Menu",
          label_visibility="collapsed", 
          options=unauth_menu
        )

        if st.session_state.unauth_selected_menu_choice == "Login":
            self.render_login_form()
        elif st.session_state.unauth_selected_menu_choice == "Register":
            self.render_register_form()
        elif st.session_state.unauth_selected_menu_choice == "Forgot Password":
            self.render_forgot_password_form()

    def render_login_form(self):
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Login"):
            if self.manager.authenticate_user(username, password):
                st.sidebar.success(f"Welcome, {st.session_state.user['name']}!")
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password")

    def render_register_form(self):
        st.session_state.email = st.sidebar.text_input("Email", value=st.session_state.email)
        st.session_state.name = st.sidebar.text_input("Name", value=st.session_state.name)
        st.session_state.password = st.sidebar.text_input(
            "Password", type="password", value=st.session_state.password
        )

        if st.sidebar.button("Register", key="register_button"):
            self.manager.register_user(st.session_state.email, st.session_state.name, st.session_state.password)

        if st.session_state.registration_success:
            st.session_state.selected_menu_choice = "Login"
            st.session_state.registration_success = False

    def render_forgot_password_form(self):
        username = st.sidebar.text_input("Username")
        if st.sidebar.button("Reset Password", key="reset_password_button"):
            self.manager.forgot_password(username)
            
    def render_change_password_form(self):
        with st.form("change_password_form", clear_on_submit=False):
            st.header("Change Password")
            old_password_input = st.text_input("Old Password", type="password", key="old_password")
            new_password_input = st.text_input("New Password", type="password", key="new_password")
            confirm_password_input = st.text_input("Confirm New Password", type="password", key="confirm_password")

            # Define a function to handle form submission
            def handle_form_submission():
                # Extract values from the inputs
                old_password = st.session_state["old_password"]
                new_password = st.session_state["new_password"]
                confirm_password = st.session_state["confirm_password"]

                self.change_password(old_password, new_password, confirm_password)
                  
            # Add an on_click function to the form submit button
            st.form_submit_button(
              "Confirm", 
              on_click=handle_form_submission,
            )

    def change_password(self, old_password, new_password, confirm_password):
        if not old_password:
            st.sidebar.error("Please enter your old password")
            return False

        if not new_password or not confirm_password:
            st.sidebar.error("Please enter a new password and confirm it")
            return False

        if new_password != confirm_password:
            st.sidebar.error("New password and confirm password do not match")
            return False

        username = st.session_state.user["email"].split("@")[0]
        if not self.manager.authenticate_user(username, old_password):
            st.sidebar.error("Invalid old password")
            return False

        # Save the new password hash to the configuration
        new_password_hash = UserUtils.hash_password(new_password)
        self.manager.config[username]["password_hash"] = new_password_hash
        self.manager.save_config(self.manager.config)
        
        st.sidebar.success("Password changed successfully!")
        return True

    def logout(self):
        st.session_state.user = None

    def authenticated_content(self):
        st.sidebar.write(f"Welcome, :green[{st.session_state.user['name']}]!")

    def run(self):
      st.title("LLMs-powered Applications")
      
      self.render_sidebar()
      
      if st.session_state.user:
        self.authenticated_content()

app = App()
app.run()

# st.write(st.session_state)