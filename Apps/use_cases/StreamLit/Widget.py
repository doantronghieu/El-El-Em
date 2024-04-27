import streamlit as st
import add_packages
from my_streamlit import utils

#*==============================================================================

STATES = {
  "COUNT": {
    "INITIAL_VALUE": 0,
  },
  "PRIVACY": {
    "INITIAL_VALUE": 'widget',
  },
  "TERMS": {
    "INITIAL_VALUE": 'widget',
  },
  "ATTENDANCE": {
    "INITIAL_VALUE": set(),
  },
  "NAME": {
    "INITIAL_VALUE": 'widget',
  },
}

utils.initialize_session_state(STATES)

#*==============================================================================

st.header("Keys distinguish widgets and access values")

col1, col2 = st.columns(2)
with col1:
  btn_ok_privacy = st.button("OK", key=STATES["PRIVACY"]["KEY"])
with col2:
  btn_ok_terms = st.button("OK", key=STATES["TERMS"]["KEY"])

#*==============================================================================

st.header("Using callback functions with forms")

form_attendance = st.form(key="form_attendance")

def take_attendance():
  if st.session_state[STATES["NAME"]["KEY"]] in st.session_state[STATES["ATTENDANCE"]["KEY"]]:
    form_attendance.info(f'{st.session_state[STATES["NAME"]["KEY"]]} has already been counted.')
  else:
    st.session_state[STATES["ATTENDANCE"]["KEY"]].add(st.session_state[STATES["NAME"]["KEY"]])

with form_attendance:
  st.text_input(label="Name", key=STATES["NAME"]["KEY"])
  st.form_submit_button("I'm here!", on_click=take_attendance)

#*==============================================================================

st.header("Changing parameters of a widget will reset it")

cols = st.columns([1, 1])

minimum = cols[0].number_input("Minimum", 1, 5)
maximum = cols[1].number_input("Maximum", 6, 10, 10)

with cols[0]:
  st.subheader("No default")
  st.slider("No key", min_value=minimum, max_value=maximum)
  st.slider("With key", min_value=minimum, max_value=maximum, key="a")
with cols[1]:
  st.subheader("With default")
  st.slider("No key", min_value=minimum, max_value=maximum, value=5)
  st.slider("With key", min_value=minimum, max_value=maximum, key="b", value=5)

#*==============================================================================

st.header("Session State")

st.write(st.session_state)
