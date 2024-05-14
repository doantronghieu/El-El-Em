import streamlit as st
import add_packages
from toolkit.streamlit import utils

#*==============================================================================

st.set_page_config(
  layout="wide",
)

#*==============================================================================

STATES = {
  "SUM": {
    "INITIAL_VALUE": '',
  },
  "N1": {
    "INITIAL_VALUE": None,
  },
  "N2": {
    "INITIAL_VALUE": None,
  },
}

utils.initialize_session_state(STATES)

#*==============================================================================

def sum():
  result = st.session_state[STATES["N1"]["KEY"]] + st.session_state[STATES["N2"]["KEY"]]
  st.session_state[STATES["SUM"]["KEY"]] = result
  return

#*==============================================================================

col1, col2, col3 = st.columns(3)

with col1:
  # Widget outside the form
  # Its value changes will trigger immediate script reruns.
  input_text_outside_form = st.text_input("Text Input (outside form)")
  # Script reruns when the user interacts with the widget
  st.write(f"Text Input (outside form) value: {input_text_outside_form}")
  
# Default values for widgets inside the form
default_text_inside_form = "Default Value"
with col2:
  # Define the form container
  my_form = st.form("my_form")
  
  # The text input widget inside this form won't trigger script reruns until the form is submitted.
  with my_form:
    # Setting default value for the text input widget inside the form
    input_text_inside_form = st.text_input(
        "Text Input (inside form)", value=default_text_inside_form
    )
    btn_submit = st.form_submit_button("Submit")
  
  # Script won't rerun until the form is submitted
  st.write(f"Text Input (inside form) value: {input_text_inside_form}")

  # If form submission button was clicked, triggering a script rerun.
  if btn_submit:
    # Execute the process after completing the form.
    st.success("Form submitted.")
    
    # Forms are containers after submitted
    my_form.success("Huarray!")

with col3:
  form_addition = st.form("addition")
  with form_addition:
    st.number_input(STATES["N1"]["KEY"], key=STATES["N1"]["KEY"])
    st.number_input(STATES["N2"]["KEY"], key=STATES["N2"]["KEY"])
    st.form_submit_button("Add", on_click=sum)
    
  col1, col2 = st.columns(2)
  
  col1.title("Sum: ")
  if isinstance(st.session_state[STATES["SUM"]["KEY"]], float):
    col2.title(f"{st.session_state[STATES['SUM']['KEY']]:.2f}")
