import streamlit as st
import yaml
from typing import Dict
from secrets import token_hex
from hashlib import sha256
from datetime import datetime, timedelta
import string
import secrets

HASH_ALGORITHM = "sha256"
SALT_LENGTH = 16

# Load configuration from users.yaml
with open("users.yaml", "r") as f:
  config = yaml.safe_load(f)

# Authentication configuration
AUTH_CONFIG = config["auth"]
COOKIE_NAME = AUTH_CONFIG["cookie"]["name"]
COOKIE_KEY = AUTH_CONFIG["cookie"]["key"]
COOKIE_EXPIRY_DAYS = AUTH_CONFIG["cookie"]["expiry_days"]
HASH_ALGORITHM = AUTH_CONFIG["passwords"]["hash_algorithm"]
SALT_LENGTH = AUTH_CONFIG["passwords"]["salt_length"]
USERS = AUTH_CONFIG["users"]
PRE_AUTHORIZED_EMAILS = AUTH_CONFIG["pre_authorized_emails"]

# Initialize session state
if "user" not in st.session_state:
  st.session_state.user = None


def hash_password(password: str, salt: str = None) -> str:
  """
  Hashes a password using the configured algorithm and salt.
  If no salt is provided, a random salt is generated.
  """
  if salt is None:
    salt = token_hex(SALT_LENGTH)
  if HASH_ALGORITHM == "sha256":
    return sha256((salt + password).encode()).hexdigest()
  else:
    raise ValueError(f"Unsupported hash algorithm: {HASH_ALGORITHM}")


def authenticate_user(username: str, password: str) -> bool:
  """
  Authenticates a user by checking if the username and password match the stored credentials.
  Returns True if authenticated, False otherwise.
  """
  user = USERS.get(username)
  if user is None:
    return False

  hashed_password = user["password_hash"].split("$")[3]
  salt = user["password_hash"].split("$")[2]
  if hash_password(password, salt) == hashed_password:
    st.session_state.user = user
    return True
  else:
    # Increment failed login attempts
    user["failed_login_attempts"] += 1
    USERS[username] = user
    return False


def register_user(email: str, name: str, password: str):
  """
  Registers a new user with the provided email, name, and password.
  """
  username = email.split("@")[0]
  if username in USERS:
    st.error("Username already exists!")
    return

  salt = token_hex(SALT_LENGTH)
  password_hash = f"{HASH_ALGORITHM}${SALT_LENGTH}${salt}${hash_password(password, salt)}"
  USERS[username] = {
    "name": name,
    "email": email,
    "password_hash": password_hash,
    "failed_login_attempts": 0,
    "locked": False,
  }

  st.success(f"User '{username}' registered successfully!")

  # Save the updated users to users.yaml
  config["auth"]["users"] = USERS
  with open("users.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)


def forgot_password(username: str):
  """
  Handles the "forgot password" functionality by generating a new password for the user.
  """
  if username not in USERS:
    st.error("Username not found!")
    return

  new_password = generate_password()
  salt = token_hex(SALT_LENGTH)
  password_hash = f"{HASH_ALGORITHM}${SALT_LENGTH}${salt}${hash_password(new_password, salt)}"
  USERS[username]["password_hash"] = password_hash

  st.success(f"New password for '{username}' is '{new_password}'")

  # Save the updated users to users.yaml
  config["auth"]["users"] = USERS
  with open("users.yaml", "w") as f:
    yaml.dump(config, f, default_flow_style=False)


def generate_password(length=12, complexity=SALT_LENGTH):
  """
  Generates a random password with the specified length and complexity.

  Args:
      length (int, optional): The length of the password. Default is 12.
      complexity (int, optional): The complexity level of the password, ranging from 1 (least complex) to 4 (most complex). Default is the same as SALT_LENGTH.

  Returns:
      str: The generated random password.

  Complexity Levels:
      1: Lowercase characters only
      2: Lowercase and uppercase characters
      3: Lowercase, uppercase, and digits
      4: Lowercase, uppercase, digits, and special characters
  """
  # Define character sets for each complexity level
  char_sets = [
    string.ascii_lowercase,
    string.ascii_letters,
    string.ascii_letters + string.digits,
    string.ascii_letters + string.digits + string.punctuation,
  ]

  # Choose the character set based on the specified complexity level
  char_set = char_sets[complexity - 1]

  # Generate the password
  password = ''.join(secrets.choice(char_set) for _ in range(length))

  return password


def check_session():
  """
  Checks if the user is authenticated by verifying the session state.
  Returns True if authenticated, False otherwise.
  """
  return st.session_state.user is not None


def set_session(user: Dict):
  """
  Sets the user session state with the user details.
  """
  st.session_state.user = user


def clear_session():
  """
  Clears the user session state.
  """
  st.session_state.user = None


def authenticated_content():
  st.header(f"Welcome, {user['name']}!")
  # Add your authenticated user content here
  


def render_sidebar():
  if check_session():
    # Render sidebar content for authenticated users
    st.sidebar.header("User Menu")
    st.sidebar.button("Logout", on_click=clear_session)
    
    # Add your authenticated user sidebar options here
      
  else:
    # Render sidebar content for unauthenticated users
    menu = ["Login", "Register", "Forgot Password"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Login":
      username = st.sidebar.text_input("Username")
      password = st.sidebar.text_input("Password", type="password")
      if st.sidebar.button("Login"):
        if authenticate_user(username, password):
          st.success(f"Welcome, {st.session_state.user['name']}!")
        else:
          st.error("Invalid username or password")

    elif choice == "Register":
      email = st.sidebar.text_input("Email")
      name = st.sidebar.text_input("Name")
      password = st.sidebar.text_input("Password", type="password")
      if st.sidebar.button("Register"):
        register_user(email, name, password)

    elif choice == "Forgot Password":
      username = st.sidebar.text_input("Username")
      if st.sidebar.button("Reset Password"):
        forgot_password(username)

    if st.sidebar.checkbox("Pre-authorized"):
      email = st.sidebar.text_input("Email")
      if email in PRE_AUTHORIZED_EMAILS:
        user = {"name": email.split("@")[0], "email": email}
        set_session(user)
        st.experimental_rerun()
      else:
        st.error("Email not pre-authorized")


# Streamlit app
st.set_page_config(layout="wide")
st.title("LLMs-powered Applications")

render_sidebar()

if check_session():
  user = st.session_state.user
  authenticated_content()