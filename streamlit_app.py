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
  y = data
  y

with st.expander('Data Visualization'):
  st.scatter_chart(data=data, x='bill_length_mm',y='body_mass_g',color = 'species')

# data preparation
with st.sidebar:
  st.header('Input Features')
  island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
  bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
  bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
  flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
  body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)
  gender = st.selectbox('Gender', ('male', 'female'))


  # Create a DataFrame for the input features
  data1 = {'island': island,
          'bill_length_mm': bill_length_mm,
          'bill_depth_mm': bill_depth_mm,
          'flipper_length_mm': flipper_length_mm,
          'body_mass_g': body_mass_g,
          'sex': gender}
  input_df = pd.DataFrame(data1, index=[0])
  input_penguins = pd.concat([input_df, X_raw], axis=0)
  
  
  
