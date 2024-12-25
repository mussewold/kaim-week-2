import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_experience_metrics(df):
    """
    Aggregate experience metrics per customer
    """
    experience_metrics = df.groupby('user_id').agg({
        'TCP DL Retrans. Vol (Bytes)': 'mean',
        'TCP UL Retrans. Vol (Bytes)': 'mean',
        'Avg RTT DL (ms)': 'mean',
        'Avg RTT UL (ms)': 'mean',
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean',
        'handset': 'first'
    }).rename(columns={
        'TCP DL Retrans. Vol (Bytes)': 'tcp_dl_retrans',
        'TCP UL Retrans. Vol (Bytes)': 'tcp_ul_retrans',
        'Avg RTT DL (ms)': 'rtt_dl',
        'Avg RTT UL (ms)': 'rtt_ul',
        'Avg Bearer TP DL (kbps)': 'throughput_dl',
        'Avg Bearer TP UL (kbps)': 'throughput_ul'
    })
    
    # Add combined metrics
    experience_metrics['tcp_retransmission'] = (
        experience_metrics['tcp_dl_retrans'] + experience_metrics['tcp_ul_retrans']
    )
    experience_metrics['avg_rtt'] = (
        experience_metrics['rtt_dl'] + experience_metrics['rtt_ul']
    ) / 2
    experience_metrics['avg_throughput'] = (
        experience_metrics['throughput_dl'] + experience_metrics['throughput_ul']
    ) / 2
    
    return experience_metrics

def analyze_network_parameter(df, parameter):
    """
    Analyze network parameter (top, bottom, most frequent values)
    """
    return {
        'top': df[parameter].nlargest(10).tolist(),
        'bottom': df[parameter].nsmallest(10).tolist(),
        'frequent': df[parameter].value_counts().head(10).index.tolist()
    }

def analyze_throughput_by_handset(df):
    """
    Analyze throughput distribution per handset type
    """
    return df.groupby('handset')['avg_throughput'].describe()

def analyze_tcp_by_handset(df):
    """
    Analyze TCP retransmission per handset type
    """
    return df.groupby('handset')['tcp_retransmission'].describe()

def plot_throughput_distribution(throughput_dist):
    """
    Plot throughput distribution by handset type
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=throughput_dist)
    plt.xticks(rotation=45)
    plt.title('Throughput Distribution by Handset Type')
    plt.tight_layout()
    return plt.gcf()

def plot_tcp_by_handset(tcp_dist):
    """
    Plot TCP retransmission by handset type
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=tcp_dist)
    plt.xticks(rotation=45)
    plt.title('TCP Retransmission by Handset Type')
    plt.tight_layout()
    return plt.gcf()

def perform_experience_clustering(df):
    """
    Perform k-means clustering on experience metrics
    """
    # Select features for clustering
    features = ['tcp_retransmission', 'avg_rtt', 'avg_throughput']
    X = df[features]
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calculate cluster statistics
    cluster_stats = df.groupby('cluster')[features].agg(['mean', 'std'])
    
    # Generate cluster descriptions
    descriptions = {
        0: "High Performance Cluster: [Add description based on results]",
        1: "Medium Performance Cluster: [Add description based on results]",
        2: "Low Performance Cluster: [Add description based on results]"
    }
    
    return {
        'data': df,
        'stats': cluster_stats,
        'descriptions': descriptions
    }

def plot_experience_clusters(df):
    """
    Plot experience cluster visualizations
    """
    fig = plt.figure(figsize=(15, 5))
    
    # Plot 1: TCP vs RTT
    plt.subplot(131)
    plt.scatter(df['tcp_retransmission'], df['avg_rtt'], 
               c=df['cluster'], cmap='viridis')
    plt.xlabel('TCP Retransmission')
    plt.ylabel('Average RTT')
    plt.title('TCP vs RTT by Cluster')

    # Plot 2: TCP vs Throughput
    plt.subplot(132)
    plt.scatter(df['tcp_retransmission'], df['avg_throughput'], 
               c=df['cluster'], cmap='viridis')
    plt.xlabel('TCP Retransmission')
    plt.ylabel('Average Throughput')
    plt.title('TCP vs Throughput by Cluster')

    # Plot 3: RTT vs Throughput
    plt.subplot(133)
    plt.scatter(df['avg_rtt'], df['avg_throughput'], 
               c=df['cluster'], cmap='viridis')
    plt.xlabel('Average RTT')
    plt.ylabel('Average Throughput')
    plt.title('RTT vs Throughput by Cluster')

    plt.tight_layout()
    return fig 