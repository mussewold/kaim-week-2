import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

def plot_distribution(df: pd.DataFrame, column: str, title: str = None):
    """Plot distribution of a column"""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column, kde=True)
    plt.title(title or f'Distribution of {column}')
    plt.show()

def plot_boxplot(df: pd.DataFrame, column: str, title: str = None):
    """Plot boxplot of a column"""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, y=column)
    plt.title(title or f'Boxplot of {column}')
    plt.show()

def plot_correlation_matrix(corr_matrix: pd.DataFrame):
    """Plot correlation matrix heatmap"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()

def plot_pca_explained_variance(pca) -> None:
    """Plot PCA explained variance ratio"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
             np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('PCA Explained Variance Ratio')
    plt.show() 