import streamlit as st
from data_utils import *

# Try to import Plotly and install if not available
try:
    import plotly.express as px
except ModuleNotFoundError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    import plotly.express as px

# Set page config for dark mode
st.set_page_config(
    page_title="AnalytiQ",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Function to display the summary as tiles
def display_summary_tiles(summary):
    """Displays the summary statistics in a tile format."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Number of Rows", summary['Number of Rows'])
    col1.metric("Number of Columns", summary['Number of Columns'])
    col2.metric("Missing Values", summary['Missing Values'])
    col2.metric("Duplicate Rows", summary['Duplicate Rows'])
    col3.metric("Memory Usage (MB)", summary['Memory Usage (MB)'])


# Function to display the column-level summary and distribution side by side
def display_column_summary(df, column):
    """Displays the summary of the selected column with distribution plots."""
    summary = column_summary(df, column)
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Data Type:", summary['Data Type'])
        st.write("Unique Values:", summary['Unique Values'])
        st.write("Missing Values:", summary['Missing Values'])
        st.write("Mean:", summary['Mean'])
        st.write("Median:", summary['Median'])
        st.write("Mode:", summary['Mode'])
        st.write("Standard Deviation:", summary['Standard Deviation'])
        st.write("Min:", summary['Min'])
        st.write("Max:", summary['Max'])
    
    with col2:
        st.subheader(f"Distribution of {column}")
        if pd.api.types.is_numeric_dtype(df[column]):
            fig = px.histogram(df, x=column, marginal="box", nbins=30, title=f'Distribution of {column}')
        else:
            fig = px.histogram(df, x=column, color=column, title=f'Distribution of {column}')
        st.plotly_chart(fig, use_container_width=True)

# Main app function
def main():
    st.title("AnalytiQ")
    
    # Define the datasets folder path
    datasets_folder = "datasets"
    
    # Fetch dataset file names
    datasets = load_datasets(datasets_folder)
    
    # Sidebar for dataset selection, limit input, and filters
    st.sidebar.header("Select Dataset and Filters")
    dataset_name = st.sidebar.selectbox("Select Dataset", datasets)
    data_limit = st.sidebar.number_input("Number of Rows to Fetch", min_value=1, value=1000, step=1000)
    
    # Load the selected dataset with a loading spinner
    if dataset_name:
        data_path = os.path.join(datasets_folder, dataset_name)
        with st.spinner(f"Loading {dataset_name}..."):
            selected_data = load_data(data_path, data_limit)
        
        # Sidebar filters
        st.sidebar.subheader("Filter Data")
        filters = {}
        for column in selected_data.columns:
            unique_vals = selected_data[column].unique()
            if len(unique_vals) < 100:  # Only show filter options if there are less than 100 unique values
                filters[column] = st.sidebar.selectbox(f"Filter by {column}", options=[None] + list(unique_vals))
        
        # Apply filters
        filtered_data = apply_filters(selected_data, filters)
        
        # Tabs for different views (e.g., Data View, Analysis, etc.)
        tabs = st.tabs(["Data View", "Other Tab 1", "Other Tab 2"])
        
        with tabs[0]:
            st.header("Data Summary")
            
            # Display summary statistics in tiles
            summary = generate_summary(filtered_data)
            display_summary_tiles(summary)
            
            # Combined accordion with two tabs for detailed statistics and column-level summary
            st.header("Detailed Analysis")
            with st.expander("View Detailed Analysis", expanded=False):
                sub_tabs = st.tabs(["Detailed Statistics", "Column-Level Summary"])
                
                with sub_tabs[0]:
                    st.subheader("Detailed Statistics")
                    st.dataframe(detailed_statistics(filtered_data), use_container_width=True)
                
                with sub_tabs[1]:
                    st.subheader("Column-Level Summary")
                    selected_column = st.selectbox("Select Column", filtered_data.columns)
                    if selected_column:
                        display_column_summary(filtered_data, selected_column)
        
        # Placeholder content for other tabs
        with tabs[1]:
            st.header("Other Analysis 1")
            st.write("Content for the second tab goes here.")
        
        with tabs[2]:
            st.header("Other Analysis 2")
            st.write("Content for the third tab goes here.")
        
        # View Data section remains constant across all tabs
        st.header("View Data")
        st.write(f"Displaying first {data_limit} rows of {dataset_name}")
        st.write(f"Data filtered by {filters}")
        st.dataframe(filtered_data, use_container_width=True)
            
if __name__ == "__main__":
    main()
