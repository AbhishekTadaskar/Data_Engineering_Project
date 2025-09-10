import streamlit as st
import pandas as pd
import pickle

# --- Load Data and Model ---
@st.cache_data
def load_data():
    """Loads and merges the three datasets."""
    try:
        df_item = pd.read_xml('df_item.xml')
        df_outlet = pd.read_xml('df_outlet.xml')
        df_sales = pd.read_xml('df_sales.xml')

        # Combine the dataframes. This assumes a common 'ID' column.
        # df_sales only has ID and sales, so we'll merge item and outlet info first.
        df_merged = pd.merge(df_item, df_outlet, on='ID', how='left')
        
        # The model was likely trained on features from both merged dataframes,
        # so we'll create a single dataframe for feature extraction.
        return df_merged
    except FileNotFoundError as e:
        st.error(f"Error loading data files: {e}. Please ensure 'df_item.xml', 'df_outlet.xml', and 'df_sales.xml' are in the same directory.")
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
        st.error(f"Error loading the model: {e}")
        return None

data = load_data()
model = load_model()

if data is not None and model is not None:
    
    # --- UI Elements ---
    st.set_page_config(
        page_title="BigMart Sales Predictor",
        page_icon="🛒",
        layout="wide"
    )

    st.title("🛒 BigMart Sales Prediction App")
    st.markdown("Use this app to predict the sales of an item at a specific outlet.")

    # Get unique values for dropdowns
    item_types = sorted(data['Item_Type'].unique())
    fat_contents = sorted(data['Item_Fat_Content'].unique())
    outlet_sizes = sorted(data['Outlet_Size'].unique())
    outlet_location_types = sorted(data['Outlet_Location_Type'].unique())
    outlet_types = sorted(data['Outlet_Type'].unique())

    # Create input columns
    col1, col2 = st.columns(2)

    with col1:
        st.header("Item Details")
        item_weight = st.number_input("Item Weight", value=12.0, min_value=0.1, max_value=25.0, step=0.1)
        item_fat_content = st.selectbox("Item Fat Content", fat_contents)
        item_visibility = st.number_input("Item Visibility", value=0.07, min_value=0.0, max_value=0.3, step=0.01)
        item_type = st.selectbox("Item Type", item_types)
        item_mrp = st.number_input("Item MRP", value=150.0, min_value=1.0, max_value=300.0, step=1.0)
    
    with col2:
        st.header("Outlet Details")
        outlet_establishment_year = st.number_input("Outlet Establishment Year", value=2000, min_value=1985, max_value=2015, step=1)
        outlet_size = st.selectbox("Outlet Size", outlet_sizes)
        outlet_location_type = st.selectbox("Outlet Location Type", outlet_location_types)
        outlet_type = st.selectbox("Outlet Type", outlet_types)

    # --- Prediction Logic ---
    if st.button("Predict Sales"):
        # Create a single row DataFrame from user inputs
        input_df = pd.DataFrame([{
            'Item_Weight': item_weight,
            'Item_Fat_Content': item_fat_content,
            'Item_Visibility': item_visibility,
            'Item_Type': item_type,
            'Item_MRP': item_mrp,
            'Outlet_Establishment_Year': outlet_establishment_year,
            'Outlet_Size': outlet_size,
            'Outlet_Location_Type': outlet_location_type,
            'Outlet_Type': outlet_type,
            'Item_Identifier': 'FDX00', # Placeholder for prediction
            'Outlet_Identifier': 'OUT000', # Placeholder for prediction
        }])

        try:
            # Make the prediction
            prediction = model.predict(input_df)
            predicted_sales = prediction[0]

            st.markdown("---")
            st.subheader("Predicted Sales")
            st.success(f"The predicted Item Outlet Sales are: ${predicted_sales:,.2f}")
            st.balloons()

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
