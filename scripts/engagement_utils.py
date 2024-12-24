import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def get_top_users_by_metric(df, metric_col, n=10):
    """Get top n users by a specific metric"""
    return df.nlargest(n, metric_col)[[metric_col]]

def perform_kmeans_clustering(df, features, n_clusters=3):
    """Perform k-means clustering on specified features"""
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    return clusters, kmeans, X_scaled

def calculate_cluster_statistics(df, cluster_col, metrics):
    """Calculate statistics for each cluster"""
    return df.groupby(cluster_col).agg({
        metric: ['min', 'max', 'mean', 'sum'] for metric in metrics
    })

def find_optimal_k(X, k_range):
    """Find optimal k using elbow method and silhouette score"""
    inertias = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    
    return inertias, silhouette_scores

def get_app_usage_metrics(df):
    """Calculate total usage metrics per application"""
    app_columns = ['social_media_total', 'google_total', 'email_total', 
                  'youtube_total', 'netflix_total', 'gaming_total', 'other_total']
    return df[app_columns].sum().sort_values(ascending=False) 