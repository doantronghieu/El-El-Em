import streamlit as st
import random
import numpy as np
import time
import add_packages
from toolkit.streamlit import utils

st.set_page_config(
  layout="wide",
)
#*==============================================================================

STATES = {
  "IS_CLICKED": {
    "INITIAL_VALUE": False,
  },
  "AVAILABLE_ITEMS": {
    "INITIAL_VALUE": ["book", "chair", "table", "lamp"], 
  },
  "IS_TOGGLE_WIDGET": {
    "INITIAL_VALUE": False,
  },
  "STAGE": {
    "INITIAL_VALUE": 0,
  },
  "NAME": {
    "INITIAL_VALUE": 'Initial Name',
  },
  "WIDGETS": {
    "INITIAL_VALUE": 0,
  },
  "EXPENSIVE_PROCESS_RESULT": {
    "INITIAL_VALUE": 0,
  },
}

utils.initialize_session_state(STATES)

#*==============================================================================

st.header("Animate and update elements")
st.write(
"""
`st.empty` containers can be written to in sequence and will always show the last thing written. They can be cleared with an additional `.empty()` method. `st.empty` hold a single element, when written, previous content was discard and displays the new element. To update a set of elements, use a plain `st.container` inside `st.empty` container and rewrite the contents of `st.container` as needed to update your app's display.

`st.dataframe`, `st.table`, and chart elements can be updated with the `.add_rows(`) method to append data.

`st.progress` elements can be updated with additional `.progress()` calls and cleared with a `.empty()` method call.

`st.status` containers have an `.update()` method for changing labels, expanded state, and status.

Toast messages can be updated with more `.toast()` calls.
"""
)

#*==============================================================================

st.header("Button behavior")

#*------------------------------------------------------------------------------

st.subheader("Display a temporary message")
# Quick button to check the validity of an entry without keeping the check displayed.

input_item = st.text_input("Type an item")

if st.button("Check availability"):
  has_item = input_item.lower() in st.session_state[STATES["AVAILABLE_ITEMS"]["KEY"]]
  if has_item:
    st.success("We have that item!")
  else:
    st.warning("We don't have that item.")

#*------------------------------------------------------------------------------

st.subheader("Stateful button")
st.write(
"""
To keep a clicked button True, set a value in st.session_state and use the 
button to update it in a callback.
"""
)

def on_click_btn():
  st.session_state[STATES["IS_CLICKED"]["KEY"]] = True

st.button("Click me", on_click=on_click_btn)

if st.session_state[STATES["IS_CLICKED"]["KEY"]]:
  st.success("Button is clicked!")

#*------------------------------------------------------------------------------

st.subheader("Toggle button")
st.write("Toggle another widget on and off.")

def on_click_btn_toggle_widget():
  st.session_state[STATES["IS_TOGGLE_WIDGET"]["KEY"]] = \
    not st.session_state[STATES["IS_TOGGLE_WIDGET"]["KEY"]]

st.button("Toggle widget", on_click=on_click_btn_toggle_widget)

if st.session_state[STATES["IS_TOGGLE_WIDGET"]["KEY"]]:
  st.success("Toggle-Widget-Button if ON")
  st.slider("NICE")
else:
  st.warning("Toggle-Widget-Button if OFF")

#*------------------------------------------------------------------------------

st.subheader("Buttons for progressing through process stages")
st.write("Use a value in st.session_state to indicate the stage of a process.")

def on_click_set_stage(i: int):
  st.session_state[STATES["STAGE"]["KEY"]] = i

if st.session_state[STATES["STAGE"]["KEY"]] == 0:
  st.button("Begin", on_click=on_click_set_stage, kwargs=dict(i=1,))
if st.session_state[STATES["STAGE"]["KEY"]] >= 1:
  input_name = st.text_input("Name", on_change=on_click_set_stage, 
                                  kwargs=dict(i=2,)
  )
if st.session_state[STATES["STAGE"]["KEY"]] >= 2:
  st.write(f"Hello {input_name}")
  input_color = st.selectbox(
    "Pick a color",
    [None, 'red', 'orange', 'green', 'blue', 'violet'],
    on_change=on_click_set_stage, kwargs=dict(i=3),
  )
  if input_color is None:
    on_click_set_stage(2)
if st.session_state[STATES["STAGE"]["KEY"]] >= 3:
  st.write(f":{input_color}[Thank you]")
  st.button("Start Over", on_click=on_click_set_stage, kwargs=dict(i=0))

#*------------------------------------------------------------------------------

st.subheader("Buttons to change st.session_state")
st.write("Modify session state in callback logic (executed before the script reruns).")

def on_click_change_name(name: str):
  st.session_state[STATES["NAME"]["KEY"]] = name

st.header(st.session_state[STATES["NAME"]["KEY"]])

cols = st.columns([1, 1])
cols[0].button("Change Name to `Jane`", on_click=on_click_change_name, 
          kwargs=dict(name="Jane"))
cols[1].button("Change Name to `John`", on_click=on_click_change_name, 
          kwargs=dict(name="John"))

st.header(st.session_state[STATES["NAME"]["KEY"]])

#*------------------------------------------------------------------------------

st.subheader("Buttons to add other widgets dynamically")

def display_widget(index):
  # Function to display a random emoji based on index
  emojis = ["😀", "😊", "😎", "🚀", "🎉", "🌟"]
  emoji = random.choice(emojis)  
  st.write(f'Widget {index+1}: {emoji}')  

def on_click_add_widget():
  # Add a new widget when button is clicked
  st.session_state[STATES["WIDGETS"]["KEY"]] += 1  # Increase the widgets counter

st.button('Add Widget', on_click=on_click_add_widget)

# Loop through the number of widgets and display them
for i in range(st.session_state[STATES["WIDGETS"]["KEY"]]):
    display_widget(i)

#*------------------------------------------------------------------------------

st.subheader("Buttons to handle expensive or file-writing processes")

def expensive_process():
  # Function to simulate a costly process
  with st.spinner('Processing...'):
    # Simulate a costly process by generating a large list of random numbers
    random_data = np.random.randint(0, 1000, size=(10000, 10))
    # Perform some operations on the random data (e.g., mean calculation)
    result = np.mean(random_data)
    time.sleep(5)  # Simulate processing time
  return result

# Button to trigger the costly process
if st.button('Run Expensive Process'):
  st.session_state[STATES["EXPENSIVE_PROCESS_RESULT"]["KEY"]] = expensive_process()

# Display the result if available
if st.session_state[STATES["EXPENSIVE_PROCESS_RESULT"]["KEY"]] is not None:
  st.success(f'Result: {st.session_state[STATES["EXPENSIVE_PROCESS_RESULT"]["KEY"]]}')

#*==============================================================================

st.header("Session State")
st.write(st.session_state)