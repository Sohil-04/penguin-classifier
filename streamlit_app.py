import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Penguin Classifier", layout="wide")

st.title('🐧 Penguin Species Classifier')
st.caption("A simple machine learning app built with Streamlit")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("penguins_cleaned.csv")

data = load_data()
x = data.drop('species', axis=1)
y_raw = data['species']

# Sidebar Inputs
st.sidebar.header("🔧 Input Features")

island = st.sidebar.selectbox('Island', ['Biscoe', 'Dream', 'Torgersen'])
sex = st.sidebar.selectbox('Sex', ['male', 'female'])
bill_length_mm = st.sidebar.slider('Bill Length (mm)', 32.1, 59.6, 43.9)
bill_depth_mm = st.sidebar.slider('Bill Depth (mm)', 13.1, 21.5, 17.2)
flipper_length_mm = st.sidebar.slider('Flipper Length (mm)', 172.0, 231.0, 201.0)
body_mass_g = st.sidebar.slider('Body Mass (g)', 2700.0, 6300.0, 4207.0)

# Optional model tuning
with st.sidebar.expander("⚙️ Model Settings", expanded=False):
    n_trees = st.slider("Number of Trees", 10, 200, 100, step=10)
    max_depth = st.slider("Max Depth", 1, 20, 5)

# Create input DataFrame
input_dict = {
    'island': island,
    'bill_length_mm': bill_length_mm,
    'bill_depth_mm': bill_depth_mm,
    'flipper_length_mm': flipper_length_mm,
    'body_mass_g': body_mass_g,
    'sex': sex
}
input_df = pd.DataFrame([input_dict])
input_all = pd.concat([input_df, x], axis=0)

# Encoding
input_encoded = pd.get_dummies(input_all, columns=['island', 'sex'])
X = input_encoded[1:]
input_row = input_encoded[:1]

# Target encoding
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
y = y_raw.map(target_mapper)

# Train model
clf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth)
clf.fit(X, y)
prediction = clf.predict(input_row)
prediction_proba = clf.predict_proba(input_row)

penguin_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
predicted_name = penguin_species[prediction][0]

# Tabs for sections
tab1, tab2, tab3 = st.tabs(["📊 Data", "📈 Visualization", "🤖 Prediction"])

with tab1:
    st.subheader("Raw Data")
    st.dataframe(data)

with tab2:
    st.subheader("Bill Length vs Body Mass")
    st.scatter_chart(data, x="bill_length_mm", y="body_mass_g", color="species")

with tab3:
    st.subheader("Predicted Species")
    st.success(f"🎉 The predicted penguin species is: **{predicted_name}**")

    st.subheader("Prediction Probabilities")
    df_proba = pd.DataFrame(prediction_proba, columns=['Adelie', 'Chinstrap', 'Gentoo'])

    st.dataframe(df_proba.style.highlight_max(axis=1, color='lightgreen'))

    with st.expander("🔍 Explanation"):
        st.write("We use a Random Forest Classifier trained on Palmer Penguins dataset.")
        st.write(f"Current Model Parameters: `n_estimators={n_trees}`, `max_depth={max_depth}`")
        st.write("Predictions are based on the input features using a voting ensemble of decision trees.")

    # Allow users to download prediction
    csv = df_proba.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Prediction Probabilities",
        data=csv,
        file_name='penguin_prediction.csv',
        mime='text/csv',
    )
