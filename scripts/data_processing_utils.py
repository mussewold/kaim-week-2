import pandas as pd
import numpy as np
import os

def load_and_preprocess_data():
    """Load and preprocess the telecom dataset from data folder"""
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to project root and then into data directory
    data_path = os.path.join(current_dir, '..', 'data', 'telecom_data.csv')
    
    # Check if file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find the data file at {data_path}. Please ensure telecom_data.csv is in the data directory.")
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Rename columns for easier handling
    df = df.rename(columns={
        'Bearer Id': 'bearer_id',
        'Start': 'start_time',
        'End': 'end_time',
        'Dur. (ms)': 'duration_ms',
        'MSISDN/Number': 'user_id',
        'IMEI': 'handset_type',
        'Handset Manufacturer': 'manufacturer',
        'Handset Type': 'handset',
        'Total UL (Bytes)': 'total_ul',
        'Total DL (Bytes)': 'total_dl',
        'Social Media DL (Bytes)': 'social_media_dl',
        'Social Media UL (Bytes)': 'social_media_ul',
        'Google DL (Bytes)': 'google_dl',
        'Google UL (Bytes)': 'google_ul',
        'Email DL (Bytes)': 'email_dl',
        'Email UL (Bytes)': 'email_ul',
        'Youtube DL (Bytes)': 'youtube_dl',
        'Youtube UL (Bytes)': 'youtube_ul',
        'Netflix DL (Bytes)': 'netflix_dl',
        'Netflix UL (Bytes)': 'netflix_ul',
        'Gaming DL (Bytes)': 'gaming_dl',
        'Gaming UL (Bytes)': 'gaming_ul',
        'Other DL (Bytes)': 'other_dl',
        'Other UL (Bytes)': 'other_ul'
    })
    
    # Basic preprocessing
    df = handle_missing_values(df)
    
    return df

def aggregate_user_metrics(df):
    """
    Aggregate metrics per user:
    - number of sessions
    - Session duration
    - Total download (DL) and upload (UL) data
    - Total data volume per application
    """
    # Basic metrics
    user_metrics = df.groupby('user_id').agg({
        'bearer_id': 'count',
        'duration_ms': 'sum',
        'total_dl': 'sum',
        'total_ul': 'sum'
    }).rename(columns={
        'bearer_id': 'total_sessions',
        'duration_ms': 'total_duration'
    })
    
    # Add total volume
    user_metrics['total_volume'] = user_metrics['total_dl'] + user_metrics['total_ul']
    
    # Aggregate application-specific data
    app_pairs = [
        ('social_media', ['social_media_dl', 'social_media_ul']),
        ('google', ['google_dl', 'google_ul']),
        ('email', ['email_dl', 'email_ul']),
        ('youtube', ['youtube_dl', 'youtube_ul']),
        ('netflix', ['netflix_dl', 'netflix_ul']),
        ('gaming', ['gaming_dl', 'gaming_ul']),
        ('other', ['other_dl', 'other_ul'])
    ]
    
    for app_name, columns in app_pairs:
        dl_col, ul_col = columns
        if dl_col in df.columns and ul_col in df.columns:
            user_metrics[f'{app_name}_total'] = df.groupby('user_id')[dl_col].sum() + \
                                              df.groupby('user_id')[ul_col].sum()
    
    return user_metrics

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    # Replace missing values with mean for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    
    # Fill categorical columns with mode
    categorical_columns = df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
    
    return df

def create_decile_classes(df, column):
    """Create decile classes based on a specific column"""
    df[f'{column}_decile'] = pd.qcut(df[column], q=10, labels=['D1', 'D2', 'D3', 'D4', 'D5', 
                                                               'D6', 'D7', 'D8', 'D9', 'D10'])
    return df 