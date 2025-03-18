import streamlit as st
import pandas as pd

def data_overview(df):
    if df is not None:
        # Basic information
        st.header("Data Overview")
        
        # Data preview with row selection
        num_rows = min(5, len(df))
        num_rows_to_display = st.slider("Number of rows to display", 5, min(100, len(df)), num_rows)
        st.dataframe(df.head(num_rows_to_display))