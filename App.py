import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import numpy as np

# ========================
# Load Model (with fix)
# ========================
@st.cache_resource(hash_funcs={dict: id})
def load_model():
    try:
        with open("bigmart_best_model.pkl", "rb") as f:
            obj = pickle.load(f)
            if isinstance(obj, tuple) and len(obj) == 2:
                model, skl_version = obj
            else:
                model, skl_version = obj, None
        return model, skl_version
    except FileNotFoundError:
        st.error("❌ Model file 'bigmart_best_model.pkl' not found!")
        return None, None

model, skl_version = load_model()

# ========================
# App Layout
# ========================
st.set_page_config(page_title="BigMart Sales Dashboard", layout="wide")
st.title("📊 BigMart Sales Prediction Dashboard")

# Sidebar inputs
st.sidebar.header("Input Features")
item_weight = st.sidebar.number_input("Item Weight", min_value=0.0, max_value=50.0, value=10.0)
item_visibility = st.sidebar.slider("Item Visibility", 0.0, 1.0, 0.1)
item_mrp = st.sidebar.number_input("Item MRP", min_value=0.0, max_value=300.0, value=100.0)
outlet_establishment = st.sidebar.number_input("Outlet Establishment Year", min_value=1980, max_value=2025, value=2000)

# ========================
# Make Prediction
# ========================
if model is not None:
    input_data = pd.DataFrame({
        "Item_Weight": [item_weight],
        "Item_Visibility": [item_visibility],
        "Item_MRP": [item_mrp],
        "Outlet_Establishment_Year": [outlet_establishment],
    })

    if st.button("Predict Sales"):
        try:
            prediction = model.predict(input_data)
            st.success(f"💰 Predicted Sales: {prediction[0]:.2f}")
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
else:
    st.warning("⚠️ No model loaded. Please upload a valid pickle file.")

# ========================
# Sample Data Preview
# ========================
st.subheader("📂 Upload Dataset for Analysis")
uploaded_file = st.file_uploader("Upload Excel/CSV file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        st.write("✅ Dataset Loaded Successfully!")
        st.dataframe(df.head())

        # Visualization
        if "Item_MRP" in df.columns and "Item_Outlet_Sales" in df.columns:
            fig = px.scatter(df, x="Item_MRP", y="Item_Outlet_Sales", trendline="ols")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Failed to read dataset: {e}")
