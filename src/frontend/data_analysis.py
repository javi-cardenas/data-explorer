import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport

def data_analysis_tabs(df):
    if df is not None:
        st.header("Data Analysis")

        # Create tabs for different EDA aspects
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Data Info", "📊 Visualizations", "🔍 Column Analysis", "🧮 Data Filters", "📝 Data Profiling"])
        
        with tab1:
            st.subheader("Dataset Information")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Total Rows:** {df.shape[0]}")
                
                # Data types
                st.subheader("Data Types")
                dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
                dtypes_df = dtypes_df.reset_index().rename(columns={"index": "Column"})
                st.dataframe(dtypes_df)
            
            with col2:
                st.write(f"**Columns:** {df.shape[1]}")
                
                # Missing values
                st.subheader("Missing Values")
                missing_df = pd.DataFrame(df.isnull().sum(), columns=["Missing Values"])
                missing_df["Percentage"] = (missing_df["Missing Values"] / len(df) * 100).round(2)
                missing_df = missing_df.reset_index().rename(columns={"index": "Column"})
                st.dataframe(missing_df)
            
            # Statistical summary
            st.subheader("Statistical Summary")
            st.write(df.describe().T)
            
            # Correlation heatmap
            st.subheader("Correlation Matrix")
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty and numeric_df.shape[1] > 1:
                fig, ax = plt.subplots(figsize=(10, 8))
                corr_matrix = numeric_df.corr()
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
                st.pyplot(fig)
            else:
                st.info("No numeric columns available for correlation analysis or only one numeric column present.")
        
        with tab2:
            st.subheader("Data Visualization")
            
            # Column selection
            column_list = df.columns.tolist()
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
            
            # Plot types
            plot_type = st.selectbox(
                "Select Plot Type",
                ["Histogram", "Bar Chart", "Scatter Plot", "Box Plot", "Line Chart", "Pie Chart", "Count Plot"]
            )
            
            if plot_type == "Histogram":
                col = st.selectbox("Select Column for Histogram", numeric_columns)
                bins = st.slider("Number of Bins", 5, 100, 30)
                fig = px.histogram(df, x=col, nbins=bins, marginal="rug")
                st.plotly_chart(fig)
                
            elif plot_type == "Bar Chart":
                if categorical_columns:
                    x_col = st.selectbox("Select X-axis (Category)", categorical_columns)
                    if numeric_columns:
                        y_col = st.selectbox("Select Y-axis (Value)", numeric_columns)
                        agg_func = st.selectbox("Aggregation Function", ["mean", "sum", "count", "min", "max"])
                        
                        # Group by and aggregate
                        grouped_df = df.groupby(x_col)[y_col].agg(agg_func).reset_index()
                        fig = px.bar(grouped_df, x=x_col, y=y_col, title=f"{agg_func.capitalize()} of {y_col} by {x_col}")
                        st.plotly_chart(fig)
                    else:
                        st.info("No numeric columns available for Y-axis.")
                else:
                    st.info("No categorical columns available for X-axis.")
                    
            elif plot_type == "Scatter Plot":
                if len(numeric_columns) >= 2:
                    x_col = st.selectbox("Select X-axis", numeric_columns)
                    y_col = st.selectbox("Select Y-axis", numeric_columns)
                    
                    color_col = None
                    if len(column_list) > 2:
                        use_color = st.checkbox("Add Color Dimension")
                        if use_color:
                            color_col = st.selectbox("Select Color Column", column_list)
                    
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
                    st.plotly_chart(fig)
                else:
                    st.info("Need at least two numeric columns for a scatter plot.")
                    
            elif plot_type == "Box Plot":
                y_col = st.selectbox("Select Column for Box Plot", numeric_columns)
                
                use_category = False
                if categorical_columns:
                    use_category = st.checkbox("Group by Category")
                
                if use_category:
                    x_col = st.selectbox("Select Grouping Category", categorical_columns)
                    fig = px.box(df, x=x_col, y=y_col)
                else:
                    fig = px.box(df, y=y_col)
                
                st.plotly_chart(fig)
                
            elif plot_type == "Line Chart":
                if "date" in df.columns.str.lower().tolist() or "time" in df.columns.str.lower().tolist():
                    possible_date_cols = [col for col in df.columns if "date" in col.lower() or "time" in col.lower()]
                    date_col = st.selectbox("Select Date/Time Column", possible_date_cols)
                    value_col = st.selectbox("Select Value Column", numeric_columns)
                    
                    # Try to convert to datetime
                    try:
                        df_copy = df.copy()
                        df_copy[date_col] = pd.to_datetime(df_copy[date_col])
                        df_copy = df_copy.sort_values(by=date_col)
                        fig = px.line(df_copy, x=date_col, y=value_col)
                        st.plotly_chart(fig)
                    except Exception as e:
                        st.error(f"Error converting to datetime: {e}")
                else:
                    st.info("No obvious date/time columns detected. Line charts work best with time series data.")
                    x_col = st.selectbox("Select X-axis", numeric_columns)
                    y_col = st.selectbox("Select Y-axis", numeric_columns)
                    fig = px.line(df.sort_values(by=x_col), x=x_col, y=y_col)
                    st.plotly_chart(fig)
                    
            elif plot_type == "Pie Chart":
                if categorical_columns:
                    col = st.selectbox("Select Category Column", categorical_columns)
                    counts = df[col].value_counts()
                    
                    # Limit to top categories if there are too many
                    max_slices = st.slider("Maximum number of slices", 3, 15, 8)
                    if len(counts) > max_slices:
                        other_count = counts.iloc[max_slices:].sum()
                        counts = counts.iloc[:max_slices]
                        counts["Other"] = other_count
                    
                    fig = px.pie(values=counts.values, names=counts.index, title=f"Distribution of {col}")
                    st.plotly_chart(fig)
                else:
                    st.info("No categorical columns available for pie chart.")
                    
            elif plot_type == "Count Plot":
                if categorical_columns:
                    col = st.selectbox("Select Category Column", categorical_columns)
                    fig = px.histogram(df, x=col, title=f"Count of {col}")
                    st.plotly_chart(fig)
                else:
                    st.info("No categorical columns available for count plot.")
        
        with tab3:
            st.subheader("Column Analysis")
            selected_col = st.selectbox("Select Column to Analyze", df.columns)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Basic Statistics**")
                if df[selected_col].dtype in [np.number]:
                    stats = pd.DataFrame({
                        "Statistic": ["Mean", "Median", "Std Dev", "Min", "Max", "Q1 (25%)", "Q3 (75%)"],
                        "Value": [
                            df[selected_col].mean(),
                            df[selected_col].median(),
                            df[selected_col].std(),
                            df[selected_col].min(),
                            df[selected_col].max(),
                            df[selected_col].quantile(0.25),
                            df[selected_col].quantile(0.75)
                        ]
                    })
                    st.dataframe(stats)
                else:
                    st.write("Non-numeric column selected. Showing value counts instead.")
            
            with col2:
                st.write("**Value Distribution**")
                if df[selected_col].dtype in [np.number]:
                    fig = px.histogram(df, x=selected_col, marginal="box")
                    st.plotly_chart(fig)
                else:
                    value_counts = df[selected_col].value_counts().reset_index()
                    value_counts.columns = [selected_col, "Count"]
                    
                    # Show both as table and chart
                    st.dataframe(value_counts)
                    
                    # Limit to top categories for visualization
                    if len(value_counts) > 10:
                        value_counts = value_counts.head(10)
                        st.info("Showing top 10 categories only in the chart.")
                    
                    fig = px.bar(value_counts, x=selected_col, y="Count")
                    st.plotly_chart(fig)
            
            # Missing values for this column
            missing_count = df[selected_col].isnull().sum()
            missing_percent = (missing_count / len(df) * 100)
            
            st.write(f"**Missing Values:** {missing_count} ({missing_percent:.2f}%)")
            
            # For numeric columns, show outlier analysis
            if df[selected_col].dtype in [np.number]:
                st.subheader("Outlier Analysis")
                
                q1 = df[selected_col].quantile(0.25)
                q3 = df[selected_col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)][selected_col]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("IQR", f"{iqr:.2f}")
                with col2:
                    st.metric("Lower Bound", f"{lower_bound:.2f}")
                with col3:
                    st.metric("Upper Bound", f"{upper_bound:.2f}")
                
                st.write(f"**Number of Outliers:** {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")
                
                if len(outliers) > 0:
                    st.write("**Sample of Outliers:**")
                    st.dataframe(outliers.head())
        
        with tab4:
            st.subheader("Data Filtering and Querying")
            
            # Select columns to show
            selected_columns = st.multiselect("Select columns to display", df.columns.tolist(), default=df.columns.tolist()[:5])
            
            # Simple filtering
            st.write("### Filter Data")
            filter_col = st.selectbox("Select column to filter", df.columns.tolist())
            
            # Different filter options based on data type
            if df[filter_col].dtype in [np.number]:
                min_val = float(df[filter_col].min())
                max_val = float(df[filter_col].max())
                
                filter_range = st.slider(
                    f"Filter range for {filter_col}",
                    min_val,
                    max_val,
                    (min_val, max_val)
                )
                
                filtered_df = df[(df[filter_col] >= filter_range[0]) & (df[filter_col] <= filter_range[1])]
            
            elif df[filter_col].dtype == "bool":
                filter_value = st.radio(f"Filter value for {filter_col}", [True, False, "Both"])
                if filter_value == "Both":
                    filtered_df = df
                else:
                    filtered_df = df[df[filter_col] == filter_value]
            
            else:  # Categorical or object
                unique_values = df[filter_col].dropna().unique().tolist()
                if len(unique_values) <= 10:
                    selected_values = st.multiselect(
                        f"Select values for {filter_col}",
                        unique_values,
                        default=unique_values
                    )
                    filtered_df = df[df[filter_col].isin(selected_values)]
                else:
                    filter_term = st.text_input(f"Enter search term for {filter_col}")
                    if filter_term:
                        filtered_df = df[df[filter_col].astype(str).str.contains(filter_term, case=False)]
                    else:
                        filtered_df = df
            
            # Additional filter based on missing values
            show_missing = st.radio("Show rows with missing values?", ["All rows", "Only rows with missing values", "Only rows with no missing values"])
            
            if show_missing == "Only rows with missing values":
                filtered_df = filtered_df[filtered_df.isnull().any(axis=1)]
            elif show_missing == "Only rows with no missing values":
                filtered_df = filtered_df[~filtered_df.isnull().any(axis=1)]
            
            # Display filtered data
            st.write(f"### Filtered Data ({len(filtered_df)} rows)")
            st.dataframe(filtered_df[selected_columns].head(100))
            
            # Option to download filtered data
            if not filtered_df.empty:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download filtered data as CSV",
                    data=csv,
                    file_name="filtered_data.csv",
                    mime="text/csv"
                )
        
        # Example usage in the profiling tab:
        with tab5:
            st.subheader("Data Profiling Report")
            st.write("Generate a comprehensive profile report of your dataset. This may take a few moments for larger datasets.")
            
            if st.button("Generate Profile Report"):
                with st.spinner("Generating profile report... This may take a while for large datasets."):
                    try:
                        # Generate profile report with minimal settings for speed
                        profile = ProfileReport(df, minimal=True)
                        st_profile_report(profile)
                    except Exception as e:
                        st.error(f"Error generating profile report: {e}")
                        st.info("For large datasets, consider using a sample of the data or the other analysis tabs.")