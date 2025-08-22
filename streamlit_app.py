import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Title and intro
st.title('Machine Learning App')
st.info('This app builds a machine learning model to classify penguin species!')

# Load and display data
with st.expander('Data'):
    st.write('***Raw Data***')
    data = pd.read_csv("penguins_cleaned.csv")
    st.dataframe(data)

    st.write('***X (Features)***')
    x = data.drop('species', axis=1)
    st.dataframe(x)

    st.write('***Y (Target)***')
    y_raw = data['species']
    st.dataframe(y_raw)

# Visualization
with st.expander('Data Visualization'):
    st.scatter_chart(data=data, x='bill_length_mm', y='body_mass_g', color='species')

# Sidebar input
with st.sidebar:
    st.header('Input Features')
    island = st.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
    gender = st.selectbox('Gender', ('male', 'female'))
    bill_length_mm = st.slider('Bill length (mm)', 32.1, 59.6, 43.9)
    bill_depth_mm = st.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
    flipper_length_mm = st.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
    body_mass_g = st.slider('Body mass (g)', 2700.0, 6300.0, 4207.0)

    # Input data as DataFrame
    input_data = {
        'island': island,
        'bill_length_mm': bill_length_mm,
        'bill_depth_mm': bill_depth_mm,
        'flipper_length_mm': flipper_length_mm,
        'body_mass_g': body_mass_g,
        'sex': gender
    }
    input_df = pd.DataFrame([input_data])

    # Combine with original data for encoding
    input_penguins = pd.concat([input_df, x], axis=0)

# Show input data
with st.expander('Input Features'):
    st.write('***Input Penguin***')
    st.dataframe(input_df)

    st.write('***Combined Data for Encoding***')
    st.dataframe(input_penguins)

# Encoding categorical variables
encode = ['island', 'sex']
df_encoded = pd.get_dummies(input_penguins, columns=encode)

# Align columns with model input (drop extra rows)
X = df_encoded[1:]
input_row = df_encoded[:1]

# Encode target variable
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
y = y_raw.map(target_mapper)

with st.expander('Data Preparation'):
    st.write('**Encoded Input (X)**')
    st.dataframe(input_row)

    st.write('**Encoded Target (y)**')
    st.dataframe(y)

# Train model
clf = RandomForestClassifier()
clf.fit(X, y)

# Predict
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

# Prepare prediction DataFrame
df_prediction_proba = pd.DataFrame(prediction_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])

# Display probabilities
st.subheader('Predicted Species Probability')
st.dataframe(
    df_prediction_proba,
    column_config={
        'Adelie': st.column_config.ProgressColumn("Adelie", format="%.2f", min_value=0, max_value=1),
        'Chinstrap': st.column_config.ProgressColumn("Chinstrap", format="%.2f", min_value=0, max_value=1),
        'Gentoo': st.column_config.ProgressColumn("Gentoo", format="%.2f", min_value=0, max_value=1)
    },
    hide_index=True
)

# Display final prediction
penguin_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(f'Predicted Penguin Species: **{penguin_species[prediction][0]}**')
