"""
    CS181DV Final Project: Interactive Data Visualization System

    Author: AIKO KATO

    Date: 05/07/2025
    
"""

import pandas as pd
import os

# Define file paths for raw input data and output summaries
RAW_PATH = os.path.join(os.path.dirname(__file__), 'Global_AI_Content_Impact_Dataset.csv')
CLEANED_PATH = os.path.join(os.path.dirname(__file__), 'cleaned_ai_content.csv')
SUMMARY_COUNTRY_PATH = os.path.join(os.path.dirname(__file__), 'summary_by_country.csv')
SUMMARY_INDUSTRY_PATH = os.path.join(os.path.dirname(__file__), 'summary_by_industry.csv')
SUMMARY_TOOL_PATH = os.path.join(os.path.dirname(__file__), 'summary_by_tool.csv')


# Load raw dataset
def load_data():
    """
    Loads the raw dataset from the CSV file specified by RAW_PATH

    Returns:
        pd.DataFrame: Raw DataFrame loaded from the CSV file

    Raises:
        RuntimeError: If the file cannot be read
    """
    try:
        df = pd.read_csv(RAW_PATH)  # Read the dataset into a DataFrame
        print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
        return df
    
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset: {e}")


# Clean and validate dataset
def clean_data(df):
    """
    Cleans the input DataFrame by:
    - Dropping rows with missing critical fields
    - Standardizing text formatting
    - Converting numeric fields
    - Dropping rows with missing numeric values

    Args:
        df (pd.DataFrame): Raw DataFrame

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Drop rows with missing values in key columns
    df.dropna(subset=['Country', 'Industry', 'Year', 'Top AI Tools Used', 'Regulation Status'], inplace=True)

    # Standardize capitalization and whitespace for text fields
    df['Country'] = df['Country'].str.strip().str.title()
    df['Industry'] = df['Industry'].str.strip().str.title()
    df['Top AI Tools Used'] = df['Top AI Tools Used'].str.strip()
    df['Regulation Status'] = df['Regulation Status'].str.strip().str.title()

    # Define numeric columns to convert to numeric dtype
    numeric_cols = [
        'AI Adoption Rate (%)',
        'AI-Generated Content Volume (TBs per year)',
        'Job Loss Due to AI (%)',
        'Revenue Increase Due to AI (%)',
        'Human-AI Collaboration Rate (%)',
        'Consumer Trust in AI (%)',
        'Market Share of AI Companies (%)'
    ]
    
    # Convert all numeric columns, coercing invalid parsing to NaN
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows that have NaN in any of the numeric columns
    df.dropna(subset=numeric_cols, inplace=True)
    
    return df


# Compute summaries
def compute_summary_stats(df):
    """
    Computes mean statistics grouped by Country, Industry, and AI Tool

    Args:
        df (pd.DataFrame): Cleaned DataFrame

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            Summaries by country, industry, and AI tool
    """
    # Group by Country and calculate the mean of numeric fields
    summary_by_country = df.groupby('Country').mean(numeric_only=True).reset_index()
    
    # Group by Industry and calculate the mean of numeric fields
    summary_by_industry = df.groupby('Industry').mean(numeric_only=True).reset_index()
    
    # Group by Top AI Tools Used and calculate the mean of numeric fields
    summary_by_tool = df.groupby('Top AI Tools Used').mean(numeric_only=True).reset_index()

    return summary_by_country, summary_by_industry, summary_by_tool


# Cache results to disk
def cache_cleaned_data(df, summary_country, summary_industry, summary_tool):
    """
    Saves cleaned and summarized datasets to CSV files

    Args:
        df (pd.DataFrame): Cleaned dataset
        summary_country (pd.DataFrame): Summary grouped by country
        summary_industry (pd.DataFrame): Summary grouped by industry
        summary_tool (pd.DataFrame): Summary grouped by AI tool
    """
    # Save cleaned dataset
    df.to_csv(CLEANED_PATH, index=False)
    
    # Save summary datasets
    summary_country.to_csv(SUMMARY_COUNTRY_PATH, index=False)
    summary_industry.to_csv(SUMMARY_INDUSTRY_PATH, index=False)
    summary_tool.to_csv(SUMMARY_TOOL_PATH, index=False)
    
    print("Cleaned and aggregated datasets cached.")


# Main preprocessing pipeline
def preprocess():
    """
    Runs the full preprocessing pipeline:
    1. Loads raw data
    2. Cleans the dataset
    3. Computes summary statistics
    4. Saves cleaned and summary files

    Returns:
        pd.DataFrame: Final cleaned DataFrame
    """
    # Step 1: Load raw CSV data into a DataFrame
    raw_df = load_data()
    
    # Step 2: Clean and validate the dataset
    clean_df = clean_data(raw_df)
    
    # Step 3: Compute summary statistics grouped by Country, Industry, and AI Tool
    summary_country, summary_industry, summary_tool = compute_summary_stats(clean_df)
    
    # Step 4: Save the cleaned dataset and summaries to disk as CSVs
    cache_cleaned_data(clean_df, summary_country, summary_industry, summary_tool)
    
    return clean_df


# Script Entry Point (Run preprocessing pipeline)
if __name__ == '__main__':
    preprocess()  # Run the full preprocessing workflow
