import streamlit as st
import pandas as pd
import pickle

# --- Load Data and Model ---
@st.cache_data
def load_data():
    """Loads and merges the datasets to get unique values for the UI."""
    try:
        df_item = pd.read_xml('df_item.xml')
        df_outlet = pd.read_xml('df_outlet.xml')
        df_sales = pd.read_xml('df_sales.xml')

        # Combine the dataframes to get all unique values for dropdowns.
        # This is for UI purposes and is not the full preprocessing logic for the model.
        df_merged = pd.merge(df_item, df_outlet, on='ID', how='left')
        
        return df_merged
    except FileNotFoundError as e:
        st.error(f"Error: A required file was not found: {e.filename}. Please ensure 'df_item.xml', 'df_outlet.xml', and 'df_sales.xml' are in the same directory.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {e}")
        return None

@st.cache_resource
def load_model():
    """Loads the pickled machine learning model."""
    try:
        with open('bigmart_best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Error: The 'bigmart_best_model.pkl' file was not found. Please ensure it is in the same directory.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the model: {e}")
        return None

data = load_data()
model = load_model()

# --- Main Application Logic ---
if data is not None and model is not None:
    
    # Set up the page configuration
    st.set_page_config(
        page_title="BigMart Sales Predictor",
        page_icon="🛒",
        layout="wide"
    )

    st.title("🛒 BigMart Sales Prediction App")
    st.markdown("Use this app to predict the sales of an item at a specific outlet based on the trained model.")

    # Get unique values for dropdowns from the merged data
    item_types = sorted(data['Item_Type'].unique())
    fat_contents = sorted(data['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}).unique())
    outlet_sizes = sorted(data['Outlet_Size'].unique())
    outlet_location_types = sorted(data['Outlet_Location_Type'].unique())
    outlet_types = sorted(data['Outlet_Type'].unique())
    
    # Identify unique identifiers for placeholders
    item_identifiers = sorted(data['Item_Identifier'].unique())
    outlet_identifiers = sorted(data['Outlet_Identifier'].unique())

    # Create input columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.header("Item Details")
        item_identifier = st.selectbox("Item Identifier", item_identifiers)
        item_weight = st.number_input("Item Weight", value=12.0, min_value=0.1, max_value=25.0, step=0.1)
        item_fat_content = st.selectbox("Item Fat Content", fat_contents)
        item_visibility = st.number_input("Item Visibility", value=0.07, min_value=0.0, max_value=0.3, step=0.01, format="%.2f")
        item_type = st.selectbox("Item Type", item_types)
        item_mrp = st.number_input("Item MRP", value=150.0, min_value=1.0, max_value=300.0, step=1.0, format="%.2f")
    
    with col2:
        st.header("Outlet Details")
        outlet_identifier = st.selectbox("Outlet Identifier", outlet_identifiers)
        outlet_establishment_year = st.number_input("Outlet Establishment Year", value=2000, min_value=1985, max_value=2015, step=1)
        outlet_size = st.selectbox("Outlet Size", outlet_sizes)
        outlet_location_type = st.selectbox("Outlet Location Type", outlet_location_types)
        outlet_type = st.selectbox("Outlet Type", outlet_types)

    # --- Prediction Logic ---
    if st.button("Predict Sales"):
        # Create a single row DataFrame from user inputs, including the required columns
        input_df = pd.DataFrame([{
            'Item_Identifier': item_identifier,
            'Item_Weight': item_weight,
            'Item_Fat_Content': item_fat_content,
            'Item_Visibility': item_visibility,
            'Item_Type': item_type,
            'Item_MRP': item_mrp,
            'Outlet_Identifier': outlet_identifier,
            'Outlet_Establishment_Year': outlet_establishment_year,
            'Outlet_Size': outlet_size,
            'Outlet_Location_Type': outlet_location_type,
            'Outlet_Type': outlet_type,
        }])
        
        try:
            # Make the prediction using the loaded model pipeline
            prediction = model.predict(input_df)
            predicted_sales = prediction[0]

            st.markdown("---")
            st.subheader("Predicted Sales")
            st.success(f"The predicted Item Outlet Sales are: **${predicted_sales:,.2f}**")
            st.balloons()

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
