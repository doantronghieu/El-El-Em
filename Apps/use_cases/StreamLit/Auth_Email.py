import streamlit as st

st.set_page_config(layout="wide")

class UserManager:
    def __init__(self):
        self.config = {}

    def authenticate_user(self, email: str) -> bool:
        if email in self.config:
            st.session_state.user = self.config[email]
            return True
        else:
            self.register_user(email, email)
            st.session_state.user = self.config[email]
            return True

    def register_user(self, email: str, name: str = None):
        if email in self.config:
            st.sidebar.error("Email already exists!")
            return
        self.config[email] = {
            "name": name if name else email,
            "email": email,
        }
        st.sidebar.success(f"User '{email}' registered successfully!")

class App:
    def __init__(self):
        self.manager = UserManager()
        self.init_session_state()

    def init_session_state(self):
        if "user" not in st.session_state:
            st.session_state.user = None

    def render_sidebar(self):
        if st.session_state.user:
            self.render_authenticated_sidebar()
        else:
            self.render_unauthenticated_sidebar()

    def render_authenticated_sidebar(self):
        st.sidebar.write(f"Welcome, :green[{st.session_state.user['name']}]!")

    def render_unauthenticated_sidebar(self):
        email = st.sidebar.text_input("Email")
        if st.sidebar.button("Login"):
            self.manager.authenticate_user(email)
            st.rerun()

    def authenticated_content(self):
        st.sidebar.write(f"Welcome, :green[{st.session_state.user['name']}]!")

    def run(self):
        st.title("LLMs-powered Applications")
        self.render_sidebar()
        if st.session_state.user:
            self.authenticated_content()

app = App()
app.run()
