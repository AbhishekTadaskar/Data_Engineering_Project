# This is a Streamlit application that loads a pre-trained machine learning model
# to predict BigMart sales based on user-provided inputs.
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings

# Suppress all warnings for cleaner output in the app
warnings.filterwarnings("ignore")

# === 1. Load the trained model ===
# The app tries to load the pre-trained model file.
# This assumes the 'bigmart_best_model.pkl' file is in the same directory.
try:
    with open("bigmart_best_model.pkl", "rb") as f:
        model, version = pickle.load(f)
    st.sidebar.success(f"Model loaded successfully (scikit-learn version: {version})")
except FileNotFoundError:
    st.error("Error: 'bigmart_best_model.pkl' not found. Please upload the file.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()


# === 2. App Title and Description ===
st.title("BigMart Sales Prediction")
st.markdown("Use the form in the sidebar to predict the sales of an item at a specific outlet.")
st.markdown("This application uses a pre-trained machine learning model to estimate the `Item_Outlet_Sales` based on item and outlet characteristics.")


# === 3. Input Form in Sidebar ===
st.sidebar.header("Input Features")
st.sidebar.markdown("Enter the details of the item and the outlet:")

# Item Features
item_weight = st.sidebar.number_input("Item Weight", min_value=1.0, max_value=25.0, value=15.5, step=0.1)
item_fat_content = st.sidebar.selectbox("Item Fat Content", ('Low Fat', 'Regular', 'reg', 'low fat', 'LF'))
item_visibility = st.sidebar.number_input("Item Visibility", min_value=0.0, max_value=0.3, value=0.07, step=0.01, help="Values will be capped at 0.3, consistent with the training data.")
item_type = st.sidebar.selectbox("Item Type", (
    'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
    'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast',
    'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads',
    'Starchy Foods', 'Others', 'Seafood'
))
item_mrp = st.sidebar.number_input("Item MRP", min_value=30.0, max_value=300.0, value=150.0, step=1.0)

# Outlet Features
outlet_establishment_year = st.sidebar.number_input("Outlet Establishment Year", min_value=1985, max_value=2025, value=2000, step=1)
outlet_size = st.sidebar.selectbox("Outlet Size", ('Small', 'Medium', 'High'))
outlet_location_type = st.sidebar.selectbox("Outlet Location Type", ('Tier 1', 'Tier 2', 'Tier 3'))
outlet_type = st.sidebar.selectbox("Outlet Type", (
    'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'
))

# === 4. Prediction Logic ===
if st.sidebar.button("Predict Sales"):
    # Create a DataFrame from user inputs.
    # The model expects 'Item_Identifier' and 'Outlet_Identifier' to be present.
    # We use a placeholder value since the one-hot encoder handles it.
    input_data = pd.DataFrame([{
        'Item_Identifier': 'FDX07', # Placeholder identifier
        'Item_Weight': item_weight,
        'Item_Fat_Content': item_fat_content,
        'Item_Visibility': item_visibility,
        'Item_Type': item_type,
        'Item_MRP': item_mrp,
        'Outlet_Identifier': 'OUT027', # Placeholder identifier
        'Outlet_Establishment_Year': outlet_establishment_year,
        'Outlet_Size': outlet_size,
        'Outlet_Location_Type': outlet_location_type,
        'Outlet_Type': outlet_type,
    }])
    
    # Feature Engineering (must match the training script)
    input_data['Outlet_Age'] = 2025 - input_data['Outlet_Establishment_Year']
    input_data.drop('Outlet_Establishment_Year', axis=1, inplace=True)
    
    # Data Cleaning (must match the training script)
    input_data['Item_Fat_Content'] = input_data['Item_Fat_Content'].replace({
        'low fat': 'Low Fat',
        'LF': 'Low Fat',
        'reg': 'Regular'
    })
    
    # Capping 'Item_Visibility' to handle outliers
    input_data['Item_Visibility'] = np.where(input_data['Item_Visibility'] > 0.3, 0.3, input_data['Item_Visibility'])
    
    # Make prediction using the loaded model
    try:
        prediction = model.predict(input_data)[0]
        # Display the result
        st.subheader("Predicted Item Outlet Sales")
        st.success(f"${prediction:,.2f}")
        st.balloons()
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# === 5. Footer ===
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and a pre-trained machine learning model.")
