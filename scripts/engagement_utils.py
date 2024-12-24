import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def get_top_users_by_metric(df, metric_col, n=10):
    """Get top n users by a specific metric"""
    return df.nlargest(n, metric_col)[[metric_col]]

def perform_kmeans_clustering(df, features, n_clusters=3):
    """Perform k-means clustering on specified features"""
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    X_scaled_df = pd.DataFrame(X_scaled, columns=features, index=df.index)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    return clusters, kmeans, X_scaled_df

def calculate_cluster_statistics(df, cluster_col, metrics):
    """Calculate statistics for each cluster"""
    return df.groupby(cluster_col).agg({
        metric: ['min', 'max', 'mean', 'sum'] for metric in metrics
    }).round(2)

def plot_cluster_visualizations(df, cluster_col='cluster'):
    """Plot cluster visualizations"""
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: Sessions vs Duration
    plt.subplot(131)
    plt.scatter(df['total_sessions'], df['total_duration'], 
               c=df[cluster_col], cmap='viridis')
    plt.xlabel('Total Sessions')
    plt.ylabel('Total Duration')
    plt.title('Sessions vs Duration by Cluster')

    # Plot 2: Sessions vs Volume
    plt.subplot(132)
    plt.scatter(df['total_sessions'], df['total_volume'], 
               c=df[cluster_col], cmap='viridis')
    plt.xlabel('Total Sessions')
    plt.ylabel('Total Volume')
    plt.title('Sessions vs Volume by Cluster')

    # Plot 3: Duration vs Volume
    plt.subplot(133)
    plt.scatter(df['total_duration'], df['total_volume'], 
               c=df[cluster_col], cmap='viridis')
    plt.xlabel('Total Duration')
    plt.ylabel('Total Volume')
    plt.title('Duration vs Volume by Cluster')

    plt.tight_layout()
    return fig

def analyze_app_usage(df, app_columns):
    """Analyze application usage and return top users per app"""
    results = {}
    
    # Get top 10 users per application
    for app in app_columns:
        results[app] = df.nlargest(10, app)[[app]]
    
    # Calculate total app usage
    app_usage = df[app_columns].sum().sort_values(ascending=False)
    top_3_apps = app_usage.head(3)
    
    return results, top_3_apps

def plot_top_apps(top_3_apps):
    """Plot top 3 most used applications"""
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(top_3_apps)), top_3_apps.values)
    plt.xticks(range(len(top_3_apps)), top_3_apps.index, rotation=45)
    plt.title('Top 3 Most Used Applications')
    plt.ylabel('Total Data Volume (bytes)')
    plt.tight_layout()
    return plt.gcf()

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

def plot_k_optimization(k_range, inertias, silhouette_scores):
    """Plot elbow curve and silhouette scores"""
    fig = plt.figure(figsize=(12, 5))
    
    # Plot 1: Elbow curve
    plt.subplot(121)
    plt.plot(k_range, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal k')

    # Plot 2: Silhouette scores
    plt.subplot(122)
    plt.plot(k_range, silhouette_scores, 'rx-')
    plt.xlabel('k')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs k')

    plt.tight_layout()
    return fig

def get_optimal_k(k_range, silhouette_scores):
    """Get optimal k based on silhouette scores"""
    return k_range[np.argmax(silhouette_scores)]