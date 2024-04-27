import streamlit as st
import add_packages
from my_streamlit import utils

#*==============================================================================

STATES = {
  "COUNT": {
    "INITIAL_VALUE": 0,
  },
  "CELSIUS": {
    "INITIAL_VALUE": 50.0,
  },
}

utils.initialize_session_state(STATES)

#*==============================================================================
st.header("Counter")

def increment_counter(value_increment):
  st.session_state[STATES["COUNT"]["KEY"]] += value_increment

def decrement_counter(value_decrement):
  st.session_state[STATES["COUNT"]["KEY"]] -= value_decrement

col1, col2 = st.columns(2)
with col1:
  value_increment = st.number_input("Increment Value", value=0, step=1)
with col2:
  value_decrement = st.number_input("Decrement Value", value=0, step=1)

col1, col2 = st.columns(2)
with col1:
  btn_increment = st.button(
    "Increment", on_click=increment_counter, args=(value_increment,)
  )
with col2:
  btn_decrement = st.button(
    "Decrement", on_click=decrement_counter, 
    kwargs=dict(value_decrement=value_decrement)
  )

#*==============================================================================

st.header("Temperature")

st.slider(
  "Temperature (Celsius)", 
  min_value=-100.0, 
  max_value=100.0,
  key=STATES["CELSIUS"]["KEY"]
)

st.header("Session States")
st.write(st.session_state)
