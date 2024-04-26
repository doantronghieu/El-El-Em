import streamlit as st
import pandas as pd
import numpy as np
import time

if st.checkbox("Table"):
  st.write("Table")
  st.write(pd.DataFrame({
    'Col 1': [1, 2, 3, 4],
    'Col 2': [10, 20, 30, 40]
  }))

if st.checkbox("Table highlighted"):
  df = pd.DataFrame(
    np.random.randn(10, 20),
    columns=('col %d' % i for i in range(20))
  )
  st.dataframe(df.style.highlight_max(axis=0))

if st.checkbox("Line Chart"):
  df_chart = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'],
  )
  st.line_chart(df_chart)

if st.checkbox("Map"):
  df_map = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4],
    columns=['lat', 'lon'],
  )
  st.map(df_map)

if st.checkbox("Slider"):
  x = st.slider("x")
  st.write(f"x^2 = {x**2}")

if st.checkbox("Text Input"):
  name = st.text_input("Your name", key="name")
  st.write(f"Your name: {st.session_state.name}")

if st.checkbox("Selectbox"):
  option = st.selectbox(
    "What color do you like?",
    ["Red", "Green", "Blue"],
  )
  st.write(f"Your favorite color: {option}")

sidebar_selectbox = st.sidebar.selectbox(
  "How would you like to be contacted?",
  ["Email", "Home Phone", "Mobile Phone"],
)
sidebar_slider = st.sidebar.slider(
  "Select a range of values",
  0.0, 100.0, (25.0, 75.0),
)

if st.checkbox("Columns"):
  col_left, col_right = st.columns(2)
  with col_left:
    st.button("Press me!")
  with col_right:
    chosen = st.radio(
      "Sorting hat",
      ("Gryffindor", "Ravenclaw", "Hufflepuff", "Slytherin"),
    )
    st.write(f"You are in {chosen} house!")
    
if st.checkbox("Progress"):
  latest_iteration = st.empty()
  bar = st.progress(0)
  
  for i in range(100):
    latest_iteration.text(f"Iteration {i + 1}")
    bar.progress(i + 1)
    time.sleep(0.1)

if st.checkbox("Session State"):
  if "counter" not in st.session_state:
    st.session_state["counter"] = 0
  
  st.session_state["counter"] += 1
  st.button("Run it again")
  st.write(f"This page has run {st.session_state['counter']} times")