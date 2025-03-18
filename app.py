import streamlit as st
import pandas as pd
import numpy as np

from src.backend.data import load_data
from src.frontend.sidebar import sidebar
from src.frontend.show_data import data_overview
from src.frontend.data_analysis import data_analysis_tabs

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")
st.title("📊 Interactive Data Explorer")
st.write("Upload your data file and explore it through interactive visualizations and analysis.")

# Initialize session state for sample data
if 'sample_data_loaded' not in st.session_state:
    st.session_state.sample_data_loaded = False

# Sidebar for upload and basic controls
uploaded_file, file_extension = sidebar()

# Main area for data exploration
if uploaded_file is not None:
    # Load the data
    df = load_data(uploaded_file, file_extension)
    
    # Data Overview
    data_overview(df)
    
    # Data Analysis
    data_analysis_tabs(df)

elif st.session_state.sample_data_loaded:
    # Use the sample data that was loaded
    df = st.session_state.df
    
    # Data Overview
    data_overview(df)

    # Data Analysis
    data_analysis_tabs(df)

else:
    st.info("👆 Please upload a data file to get started with your exploratory data analysis.")
    
    # Sample data option
    if st.button("Load Sample Data"):
        # Create sample dataset
        df = pd.DataFrame({
            'Date': pd.date_range(start='1/1/2023', periods=100),
            'Category': np.random.choice(['A', 'B', 'C', 'D'], 100),
            'Value': np.random.randn(100) * 100 + 500,
            'Growth': np.random.uniform(0, 0.3, 100),
            'Active': np.random.choice([True, False], 100, p=[0.8, 0.2])
        })
    
        # Cache the dataframe
        st.session_state.df = df
        st.session_state.sample_data_loaded = True
        st.rerun()  # Rerun the app with sample data loaded

# Add a break
st.markdown("---")
