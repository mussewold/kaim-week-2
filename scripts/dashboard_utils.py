import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_overview_page():
    """Create the User Overview Analysis page"""
    st.title("User Overview Analysis")
    
    # Load data
    df = load_and_preprocess_data()
    user_metrics = aggregate_user_metrics(df)
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Top Handsets Plot
        fig_handsets = px.bar(
            get_top_handsets(df),
            title="Top 10 Handsets",
            labels={"value": "Count", "index": "Handset"}
        )
        st.plotly_chart(fig_handsets)
        
    with col2:
        # Top Manufacturers Plot
        fig_manufacturers = px.pie(
            get_top_manufacturers(df),
            values="count",
            names="manufacturer",
            title="Top Manufacturers Distribution"
        )
        st.plotly_chart(fig_manufacturers)
    
    # Application Usage Heatmap
    app_corr = compute_correlation_matrix(user_metrics, [
        'social_media_total', 'google_total', 'email_total',
        'youtube_total', 'netflix_total', 'gaming_total', 'other_total'
    ])
    fig_heatmap = px.imshow(
        app_corr,
        title="Application Usage Correlation",
        labels=dict(color="Correlation")
    )
    st.plotly_chart(fig_heatmap)

def create_engagement_page():
    """Create the User Engagement Analysis page"""
    st.title("User Engagement Analysis")
    
    # Load data
    df = load_and_preprocess_data()
    user_metrics = aggregate_user_metrics(df)
    
    # Engagement Metrics Distribution
    fig_metrics = make_subplots(rows=1, cols=3, 
                               subplot_titles=("Sessions", "Duration", "Volume"))
    
    fig_metrics.add_trace(
        go.Histogram(x=user_metrics['total_sessions'], name="Sessions"),
        row=1, col=1
    )
    fig_metrics.add_trace(
        go.Histogram(x=user_metrics['total_duration'], name="Duration"),
        row=1, col=2
    )
    fig_metrics.add_trace(
        go.Histogram(x=user_metrics['total_volume'], name="Volume"),
        row=1, col=3
    )
    
    fig_metrics.update_layout(height=400, title="Engagement Metrics Distribution")
    st.plotly_chart(fig_metrics)
    
    # Engagement Clusters
    fig_clusters = px.scatter(
        user_metrics,
        x='total_sessions',
        y='total_volume',
        color='cluster',
        title="User Engagement Clusters"
    )
    st.plotly_chart(fig_clusters)

def create_experience_page():
    """Create the Experience Analysis page"""
    st.title("Experience Analysis")
    
    # Load data
    df = load_and_preprocess_data()
    experience_metrics = aggregate_experience_metrics(df)
    
    # Network Performance Metrics
    fig_network = make_subplots(rows=1, cols=3,
                               subplot_titles=("TCP Retransmission", "RTT", "Throughput"))
    
    fig_network.add_trace(
        go.Box(y=experience_metrics['tcp_retransmission'], name="TCP"),
        row=1, col=1
    )
    fig_network.add_trace(
        go.Box(y=experience_metrics['avg_rtt'], name="RTT"),
        row=1, col=2
    )
    fig_network.add_trace(
        go.Box(y=experience_metrics['avg_throughput'], name="Throughput"),
        row=1, col=3
    )
    
    fig_network.update_layout(height=400, title="Network Performance Metrics")
    st.plotly_chart(fig_network)
    
    # Experience by Handset
    fig_handset = px.box(
        df,
        x='handset',
        y='avg_throughput',
        title="Throughput Distribution by Handset"
    )
    st.plotly_chart(fig_handset)

def create_satisfaction_page():
    """Create the Satisfaction Analysis page"""
    st.title("Satisfaction Analysis")
    
    # Load data
    satisfaction_df = load_satisfaction_data()  # You'll need to create this function
    
    # Satisfaction Score Distribution
    fig_satisfaction = px.histogram(
        satisfaction_df,
        x='satisfaction_score',
        title="Satisfaction Score Distribution"
    )
    st.plotly_chart(fig_satisfaction)
    
    # Satisfaction vs Engagement/Experience
    fig_scatter = px.scatter(
        satisfaction_df,
        x='engagement_score',
        y='experience_score',
        color='satisfaction_score',
        title="Satisfaction Analysis"
    )
    st.plotly_chart(fig_scatter)
    
    # Satisfaction Clusters
    fig_clusters = px.scatter(
        satisfaction_df,
        x='engagement_score',
        y='experience_score',
        color='satisfaction_cluster',
        title="Satisfaction Clusters"
    )
    st.plotly_chart(fig_clusters) 