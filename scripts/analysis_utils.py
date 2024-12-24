import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_top_handsets(df, n=10):
    """Get top n handsets"""
    return df['handset'].value_counts().head(n)

def get_top_manufacturers(df, n=3):
    """Get top n manufacturers"""
    return df['manufacturer'].value_counts().head(n)

def get_top_handsets_per_manufacturer(df, manufacturers, n=5):
    """Get top n handsets for each manufacturer"""
    results = {}
    for manufacturer in manufacturers:
        mask = df['manufacturer'] == manufacturer
        results[manufacturer] = df[mask]['handset'].value_counts().head(n)
    return results

def compute_correlation_matrix(df, columns):
    """Compute correlation matrix for specified columns"""
    return df[columns].corr()

def compute_app_usage_metrics(df):
    """Compute application usage metrics"""
    app_metrics = pd.DataFrame()
    
    apps = ['social_media', 'google', 'email', 'youtube', 'netflix', 'gaming', 'other']
    
    for app in apps:
        dl_col = f'{app}_dl'
        ul_col = f'{app}_ul'
        if dl_col in df.columns and ul_col in df.columns:
            app_metrics[f'{app}_total'] = df[dl_col] + df[ul_col]
    
    return app_metrics

def perform_pca(df, columns, n_components=2):
    """Perform PCA on specified columns"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[columns])
    
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(scaled_data)
    
    return pca, pca_result 