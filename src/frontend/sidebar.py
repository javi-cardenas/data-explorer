import streamlit as st

def sidebar():
    # Sidebar for upload and basic controls
    with st.sidebar:
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "json"])
        
        file_extension = ""
        if uploaded_file is not None:
            # file_details = {
            #     "Filename": uploaded_file.name,
            #     "File size": f"{uploaded_file.size / 1024:.2f} KB",
            #     "File type": uploaded_file.type
            # }
            
            # st.write("### File Details:")
            # for key, value in file_details.items():
            #     st.write(f"**{key}:** {value}")
            
            file_extension = uploaded_file.name.split(".")[-1].lower()

        return uploaded_file, file_extension
