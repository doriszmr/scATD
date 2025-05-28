# -*- coding: utf-8 -*-
"""
Feature Importance Value Scatter Plot Analysis
This script creates scatter plots and histograms for feature importance visualization,
helping to understand the distribution and ranking of feature contributions.

Created for single-cell RNA-seq data analysis
"""

import os
import pandas as pd
import numpy as np
import re
from protloc_mex1.SHAP_plus import SHAP_importance_sum
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Set up and use command-line arguments for feature importance visualization.")

# Add command line arguments
parser.add_argument("--open_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/in', help="Path to open input data.")
parser.add_argument("--save_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/output/VAE_inference', help="Path to save output data.")
parser.add_argument("--key_name", type=str, default='feature', help="Keyword for identifying target files.")
parser.add_argument("--color", type=str, default='m', help="Color for plots (default: magenta).")

# Parse the arguments
args = parser.parse_args()

def process_data(keyword):
    """
    Process feature importance data from CSV files based on keyword matching.
    
    Parameters:
    -----------
    keyword : str
        Keyword to search for in file names
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with feature importance values
    """
    # Find file names containing a specific keyword and remove the '.csv' extension
    file_names = SHAP_importance_sum.str_contain(f'{keyword}_importance', names)
    file_names = [name.replace('.csv', '') for name in file_names]

    # Assume there's only one element in file_names
    file_name = file_names[0]

    # Create the full file path
    file_path = os.path.join(open_path, f'{file_name}.csv')

    # Read the CSV file using pandas
    df = pd.read_csv(file_path)

    # Set the 'feature_name' column as the index
    df.set_index('feature_name', inplace=True)

    # Calculate feature importance as sum of each row
    df.loc[:, 'feature importance'] = df.sum(axis=1)

    return df

def plot_histogram(df, title, bins_num=50):
    """
    Create histogram plot for feature importance distribution.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing feature importance data
    title : str
        Title for the plot
    bins_num : int
        Number of bins for the histogram
    """
    plt.figure(figsize=(5, 5))

    # Select data from the 'feature importance' column
    data = df.loc[:, 'feature importance']

    # Plot the histogram
    plt.hist(data, bins=bins_num, alpha=1, color=args.color)
    plt.title(title)
    plt.xlabel('Total Feature Importance')
    plt.ylabel('Number of Features')

    # Set PDF font parameters
    plt.rcParams['pdf.fonttype'] = 42

    # Save the image
    plt.savefig(f'{save_path}/{title}_histogram.png', dpi=1000)  # Save as PNG with high DPI
    plt.savefig(f'{save_path}/{title}_histogram.pdf')  # Save as PDF
    plt.close()  # Close the plot window

def plot_scatter_chart(df, title):
    """
    Create scatter plot for feature importance ranking.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing feature importance data
    title : str
        Title for the plot
    """
    plt.figure(figsize=(5, 5))

    # Sort the values of the 'feature importance' column and get the ranks
    data = df.loc[:, 'feature importance'].sort_values(ascending=False)
    ranks = np.arange(1, len(data) + 1)

    # Calculate log10-transformed values
    log_values = np.log10(data)

    # Plot the scatter chart
    plt.scatter(ranks, log_values, color=args.color, s=2)
    plt.title(title)
    plt.xlabel('Feature Rank')
    plt.ylabel('log10(Feature Importance)')

    # Set PDF font parameters
    plt.rcParams['pdf.fonttype'] = 42

    # Save the image
    plt.savefig(f'{save_path}/{title}_scatter_chart.png', dpi=1000)  # Save as PNG with high DPI
    plt.savefig(f'{save_path}/{title}_scatter_chart.pdf')  # Save as PDF
    plt.close()  # Close the plot window

# Main execution
# This section processes the feature importance analysis results.
# Change the 'open_path' variable to process results from different methods (RF, DNN, DNN_IG)
open_path = args.open_path
save_path = args.save_path

# Check if path exists, create if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

names = os.listdir(open_path)

# Read the target file based on the file keyword
df_train = process_data(args.key_name)

# Plot histograms and scatter charts for feature importance analysis
plot_histogram(df_train, 'feature importance rank')
plot_scatter_chart(df_train, 'feature importance rank')

print("Feature importance visualization completed successfully!")
print(f"Plots saved to: {save_path}")
