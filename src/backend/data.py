import streamlit as st
import pandas as pd

# Function to load data in chunks for large files
@st.cache_data
def load_data(file, file_type):
    try:
        if file_type == "csv":
            return pd.read_csv(file)
        elif file_type in ["xlsx", "xls"]:
            return pd.read_excel(file)
        elif file_type == "json":
            return pd.read_json(file)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None