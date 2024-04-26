import streamlit as st
import pandas as pd
import numpy as np

st.title("Uber pickups in NYC")

# Fetch the Uber dataset for New York City pickups and drop-offs
DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
  """
  Load_data function downloads data, puts it in a Pandas dataframe, and
  converts date column from text to datetime. Accepts parameter (nrows) for 
  number of rows to load.
  """
  data = pd.read_csv(DATA_URL, nrows=nrows)
  lowercase = lambda x: str(x).lower()
  data.rename(lowercase, axis='columns', inplace=True)
  data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
  return data

data_load_state = st.text("Loading data ...")
data = load_data(nrows=10000)
data_load_state.text("Done (cached).")

# Show/hide raw data table.
if st.checkbox("Raw data"):
  st.subheader("Raw data")
  st.write(data)

# Create histogram to reveal Uber's peak hours in NYC.
st.subheader("Number of pickups by hour")
# Generate a NumPy histogram for pickup times binned by hour.
hist_values = np.histogram(
  data[DATE_COLUMN].dt.hour, bins=24, range=(0, 24)
)[0]
st.bar_chart(hist_values)

# Plot data on a map.
# Show pickup concentration at 17:00, overlay the data on a map of New York City.
st.subheader("Map of all pickups")
# Filter results. Let a reader dynamically filter the data in real time
hour_to_filter = st.slider('hour', 0, 23, 17)  # min: 0h, max: 23h, default: 17h
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]
st.subheader(f'Map of all pickups at {hour_to_filter}:00')
st.map(filtered_data)

# Toggle data





