import streamlit as st
import pandas as pd 
st.title('machine learning app ')

st.info('This app build machine learning model !')

with st.expander('Data'):
  st.write('***Raw Data***')
  data = pd.read_csv("penguins_cleaned.csv")
  data

  st.write('***X***')
  x = data.drop('species',axis = 1)
  x

  st.write('***Y***')
  y = data.species
  y
