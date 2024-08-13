import pandas as pd
import os

# Function to load datasets
def load_datasets(folder_path):
    """Loads CSV file names from the specified folder."""
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    return files

# Function to load selected dataset
def load_data(file_path, limit=None):
    """Loads data from the selected CSV file and applies a row limit."""
    data = pd.read_csv(file_path)
    if limit:
        data = data.head(limit)
    return data

# Function to apply filters to the dataset
def apply_filters(df, filters):
    """Applies the user-selected filters to the DataFrame."""
    for col, val in filters.items():
        if val:
            df = df[df[col] == val]
    return df

# Function to generate a summary of the dataset
def generate_summary(df):
    """Generates a summary of the DataFrame."""
    summary = {
        'Number of Rows': len(df),
        'Number of Columns': len(df.columns),
        'Missing Values': df.isnull().sum().sum(),
        'Duplicate Rows': df.duplicated().sum(),
        'Memory Usage (MB)': round(df.memory_usage(deep=True).sum() / (1024**2), 2)
    }
    return summary

# Function to display detailed statistics for each column
def detailed_statistics(df):
    """Displays detailed statistics for each column."""
    return df.describe(include='all')

# Function to generate a column-level summary
def column_summary(df, col):
    """Generates a detailed summary for a single column."""
    summary = {
        'Data Type': df[col].dtype,
        'Unique Values': df[col].nunique(),
        'Missing Values': df[col].isnull().sum(),
        'Mean': df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
        'Median': df[col].median() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
        'Mode': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
        'Standard Deviation': df[col].std() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
        'Min': df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
        'Max': df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A',
    }
    return summary
