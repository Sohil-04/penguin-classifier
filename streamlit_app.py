import streamlit as st
import pandas as pd 
st.title('machine learning app ')

st.info('This app build machine learning model !')

with st.expander('Data'):
  st.write('***Raw Data***')
  data = pd.read_csv("penguins_cleaned.csv")
  data
