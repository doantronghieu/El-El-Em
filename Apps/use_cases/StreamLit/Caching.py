import requests
import streamlit as st
import pandas as pd
import numpy as np
from transformers import pipeline
from pydantic import BaseModel

#*==============================================================================

@st.cache_data
def long_running_fn(param1, param2):
  return ...

#*==============================================================================
# Static elements
st.subheader("Load Data")
@st.cache_data
def load_data(url):
  df = pd.read_csv(url)
  st.success("Success.")
  return df
  
if st.checkbox("Load Data"):
  url = "https://github.com/plotly/datasets/raw/master/uber-rides-data1.csv"
  df = load_data(url)
  st.dataframe(df)

  st.button("Rerun")

#*==============================================================================

# DataFrame transformations
@st.cache_data
def transform(df):
  df = df.filter(items=['one', 'three'])
  df = df.apply(np.sum, axis=0)
  return df

# Array computations
@st.cache_data
def add(arr1, arr2):
  return arr1 + arr2

# Database queries
"""
connection = database.connect()

@st.cache_data
def query():
    return pd.read_sql_query("SELECT * from table", connection)
"""

# API requests
@st.cache_data
def api_call():
  response = requests.get("https://jsonplaceholder.typicode.com/posts/1")
  return response.json()

# Running machine learning models for inference.
@st.cache_data
def run_model(model, inputs):
  return model(inputs)

#*==============================================================================

# Load the model once and use the same object for all users and sessions.
@st.cache_resource
def load_model():
  return pipeline("sentiment-analysis")

if st.checkbox("Model"):
  model = load_model()

  query = st.text_input("Your query", value="I love Streamlit! 🎈")
  if query:
    result = model(query)[0]  # 👈 Classify the query text
    st.write(result)

# Database connections
"""
@st.cache_resource
def init_connection():
  host = "hh-pgsql-public.ebi.ac.uk"
  database = "pfmegrnargs"
  user = "reader"
  password = "NWDMCE5xdipIjRrp"
  return psycopg2.connect(host=host, database=database, user=user, password=password)

conn = init_connection()
"""

#*==============================================================================

# Controlling cache size and duration
# The time-to-live (TTL) parameter
# The max_entries parameter
# Customizing spinner
# Excluding parameters
@st.cache_data(ttl=3600, max_entries=20, show_spinner=True)
def fetch_data(_db_connection, num_rows):  # 👈 Don't hash _db_connection
  data = _db_connection.fetch(num_rows)
  return data

#*==============================================================================

# The hash_funcs parameter

# Hashing a custom class
class MyCustomClass:
  def __init__(self, initial_score: int):
    self.my_score = initial_score
  
  @st.cache_data(
    hash_funcs={"__main__.MyCustomClass": lambda x: hash(x.my_score)}
  )
  def multiply_score(self, multiplier: int) -> int:
    return self.my_score * multiplier

#*------------------------------------------------------------------------------
# Hashing a Pydantic model
class Person(BaseModel):
  name: str

@st.cache_data(hash_funcs={Person: lambda p: p.name})
def identity(person: Person):
  return person