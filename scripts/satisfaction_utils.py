import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import mysql.connector
from sqlalchemy import create_engine

def calculate_euclidean_distance(point, cluster_center):
    """Calculate Euclidean distance between a point and cluster center"""
    return np.sqrt(np.sum((point - cluster_center) ** 2))

def calculate_engagement_scores(df, engagement_features, kmeans_model):
    """
    Calculate engagement scores based on distance from least engaged cluster
    """
    # Get the least engaged cluster center
    cluster_centers = kmeans_model.cluster_centers_
    least_engaged_center = cluster_centers[0]  # Assuming cluster 0 is least engaged
    
    # Calculate distances for each user
    distances = euclidean_distances(df[engagement_features], [least_engaged_center])
    
    # Normalize scores to 0-100 range
    scores = ((distances - distances.min()) / (distances.max() - distances.min())) * 100
    
    return scores.flatten()

def calculate_experience_scores(df, experience_features, kmeans_model):
    """
    Calculate experience scores based on distance from worst experience cluster
    """
    # Get the worst experience cluster center
    cluster_centers = kmeans_model.cluster_centers_
    worst_experience_center = cluster_centers[0]  # Assuming cluster 0 is worst experience
    
    # Calculate distances for each user
    distances = euclidean_distances(df[experience_features], [worst_experience_center])
    
    # Normalize scores to 0-100 range
    scores = ((distances - distances.min()) / (distances.max() - distances.min())) * 100
    
    return scores.flatten()

def calculate_satisfaction_scores(engagement_scores, experience_scores):
    """Calculate satisfaction scores as average of engagement and experience scores"""
    return (engagement_scores + experience_scores) / 2

def train_satisfaction_model(X, y, model_type='random_forest'):
    """Train a regression model to predict satisfaction scores"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
    
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    return model, train_score, test_score, X_test, y_test

def plot_satisfaction_clusters(df):
    """Plot satisfaction clusters"""
    plt.figure(figsize=(10, 6))
    plt.scatter(df['engagement_score'], df['experience_score'], 
               c=df['satisfaction_cluster'], cmap='viridis')
    plt.xlabel('Engagement Score')
    plt.ylabel('Experience Score')
    plt.title('Satisfaction Clusters')
    plt.colorbar(label='Cluster')
    return plt.gcf()

def export_to_mysql(df, table_name, connection_params):
    """Export DataFrame to MySQL database"""
    try:
        # Create SQLAlchemy engine
        engine = create_engine(
            f"mysql+mysqlconnector://{connection_params['user']}:{connection_params['password']}@"
            f"{connection_params['host']}:{connection_params['port']}/{connection_params['database']}"
        )
        
        # Export DataFrame to MySQL
        df.to_sql(table_name, engine, if_exists='replace', index=True)
        
        return True
    except Exception as e:
        print(f"Error exporting to MySQL: {str(e)}")
        return False

def plot_model_performance(y_test, y_pred):
    """Plot actual vs predicted satisfaction scores"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Satisfaction Score')
    plt.ylabel('Predicted Satisfaction Score')
    plt.title('Model Performance: Actual vs Predicted Satisfaction Scores')
    return plt.gcf() 