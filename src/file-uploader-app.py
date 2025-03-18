import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="Real Intent EDA", page_icon="📄")

st.title("Real Intent EDA")
st.write("Upload a file for Exploratory Data Analysis.")

uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "txt"])

if uploaded_file is not None:
    st.success("File successfully uploaded!")
    
    # # Display file details
    # file_details = {
    #     "Filename": uploaded_file.name,
    #     "File size": f"{uploaded_file.size} MB",
    #     "File type": uploaded_file.type
    # }
    # st.write("### File Details:")
    # for key, value in file_details.items():
    #     st.write(f"**{key}:** {value}")
    
    # Display file content based on file type
    try:
        # For CSV files
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            st.write("### First 5 rows of data:")
            st.dataframe(df.head())
            
        # For Excel files
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
            st.write("### First 5 rows of data:")
            st.dataframe(df.head())
            
        # For Text files
        elif uploaded_file.name.endswith('.txt'):
            # Try to parse as CSV first
            try:
                df = pd.read_csv(uploaded_file, sep=None, engine='python')
                st.write("### First 5 rows of data (parsed as tabular):")
                st.dataframe(df.head())
            except:
                # If CSV parsing fails, show as raw text
                stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
                string_data = stringio.read()
                st.write("### First few lines of text:")
                st.text(string_data[:1000] + ("..." if len(string_data) > 1000 else ""))
        
        # Additional file statistics
        if 'df' in locals():
            st.write("### File Statistics:")
            st.write(f"**Total rows:** {len(df)}")
            st.write(f"**Total columns:** {len(df.columns)}")
            st.write("**Column names:**")
            st.write(", ".join(df.columns.tolist()))
            
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.write("Please make sure the file is properly formatted.")

else:
    st.info("👆 Please upload a file to get started.")

# Add a break
st.markdown("---")