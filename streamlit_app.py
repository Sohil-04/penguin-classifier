import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Penguin Classifier", layout="wide")

st.title("🐧 Penguin Species Classifier")
st.caption("A simple ML app built with Streamlit to predict penguin species.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("penguins_cleaned.csv")

try:
    data = load_data()
except FileNotFoundError:
    st.error("❌ Dataset 'penguins_cleaned.csv' not found. Please add it to the app folder.")
    st.stop()

# Features & target
X_raw = data.drop("species", axis=1)
y_raw = data["species"]

# Sidebar Inputs
st.sidebar.header("🔧 Input Features")

island = st.sidebar.selectbox("Island", data["island"].unique())
sex = st.sidebar.selectbox("Sex", data["sex"].unique())
bill_length_mm = st.sidebar.slider("Bill Length (mm)", float(data.bill_length_mm.min()), float(data.bill_length_mm.max()), float(data.bill_length_mm.mean()))
bill_depth_mm = st.sidebar.slider("Bill Depth (mm)", float(data.bill_depth_mm.min()), float(data.bill_depth_mm.max()), float(data.bill_depth_mm.mean()))
flipper_length_mm = st.sidebar.slider("Flipper Length (mm)", float(data.flipper_length_mm.min()), float(data.flipper_length_mm.max()), float(data.flipper_length_mm.mean()))
body_mass_g = st.sidebar.slider("Body Mass (g)", float(data.body_mass_g.min()), float(data.body_mass_g.max()), float(data.body_mass_g.mean()))

with st.sidebar.expander("⚙️ Model Settings", expanded=False):
    n_trees = st.slider("Number of Trees", 10, 200, 100, step=10)
    max_depth = st.slider("Max Depth", 1, 20, 5)

# User input as dataframe
input_df = pd.DataFrame([{
    "island": island,
    "bill_length_mm": bill_length_mm,
    "bill_depth_mm": bill_depth_mm,
    "flipper_length_mm": flipper_length_mm,
    "body_mass_g": body_mass_g,
    "sex": sex
}])

# Combine input with training features for consistent encoding
full_df = pd.concat([input_df, X_raw], axis=0)
full_encoded = pd.get_dummies(full_df, columns=["island", "sex"])
X = full_encoded.iloc[1:]
input_encoded = full_encoded.iloc[:1]

# Encode target
target_mapper = {"Adelie": 0, "Chinstrap": 1, "Gentoo": 2}
y = y_raw.map(target_mapper)

# Cache model training
@st.cache_resource
def train_model(X, y, n_trees, max_depth):
    clf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth, random_state=42)
    clf.fit(X, y)
    return clf

clf = train_model(X, y, n_trees, max_depth)

# Predictions
prediction = clf.predict(input_encoded)
prediction_proba = clf.predict_proba(input_encoded)

penguin_species = np.array(list(target_mapper.keys()))
predicted_name = penguin_species[prediction][0]

# Tabs
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
    df_proba = pd.DataFrame(prediction_proba, columns=penguin_species)
    st.bar_chart(df_proba.T)

    with st.expander("🔍 Explanation"):
        st.write("We use a Random Forest Classifier trained on the Palmer Penguins dataset.")
        st.json({"n_estimators": n_trees, "max_depth": max_depth})

    csv = df_proba.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Prediction Probabilities",
        data=csv,
        file_name="penguin_prediction.csv",
        mime="text/csv",
    )
