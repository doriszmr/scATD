# -*- coding: utf-8 -*-
"""
Feature Importance MinMax Score SHAP Force Plot Analysis
This script creates force plots for SHAP feature importance analysis,
visualizing positive and negative contributions of features to model predictions.

Created for single-cell RNA-seq data analysis
"""

import os
import pandas as pd
import numpy as np
import re
from protloc_mex1.SHAP_plus import SHAP_importance_sum, FeaturePlot
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Set up and use command-line arguments for SHAP force plot analysis.")

# Add command line arguments
parser.add_argument("--open_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/in', help="Path to open input data.")
parser.add_argument("--save_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/output/VAE_inference', help="Path to save output data.")
parser.add_argument("--key_name", type=str, default='feature', help="Keyword for identifying target files.")
parser.add_argument("--positive_color", type=str, default='#FFB6C1', help="Color for positive SHAP values.")
parser.add_argument("--negative_color", type=str, default='#ADD8E6', help="Color for negative SHAP values.")

# Parse the arguments
args = parser.parse_args()

def process_data(keyword):
    """
    Process SHAP values data from CSV files based on keyword matching.
    
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
    file_names = SHAP_importance_sum.str_contain(f'{keyword}_shap_values', names)
    file_names = [name.replace('.csv', '') for name in file_names]

    # Assume there's only one element in file_names
    file_name = file_names[0]

    # Create the full file path
    file_path = os.path.join(open_path, f'{file_name}.csv')

    # Read the CSV file using pandas
    df = pd.read_csv(file_path, index_col=0, header=0)

    # Set the 'feature_name' column as the index
    df.set_index('feature_name', inplace=True)

    # Calculate feature importance as sum of each row
    df.loc[:, 'feature importance'] = df.sum(axis=1)

    return df

class FeaturePlotSource:
    """
    Base class for feature plotting functionality.
    """
    def __init__(self, X_data, shapley_data_all):
        self.X_data = X_data
        self.shapley_data_all = shapley_data_all

    def plot(self):
        """Base plotting method to be implemented by subclasses."""
        pass

class LocalAnalysisPlot(FeaturePlotSource):
    """
    Class for creating local analysis plots with SHAP force plots.
    Visualizes feature contributions for individual samples.
    """
    def __init__(self, X_data, shapley_data_all, probability_data):
        super().__init__(X_data, shapley_data_all)
        self.probability_data = probability_data

    def plot(self, protein_indicate, save_path, file_name='human', plot_size=(10, 10), png_plot=True,
             feature_remain=5, positive_color='#FFB6C1', negative_color='#ADD8E6'):
        """
        Create SHAP force plots for specified samples.
        
        Parameters:
        -----------
        protein_indicate : list
            List of sample identifiers to create plots for
        save_path : str
            Directory path to save the plots
        file_name : str
            Base name for output files
        plot_size : tuple
            Figure size as (width, height)
        png_plot : bool
            Whether to save PNG version in addition to PDF
        feature_remain : int
            Number of top features to display
        positive_color : str
            Color for positive SHAP values
        negative_color : str
            Color for negative SHAP values
        """
        
        for value in list(self.shapley_data_all.keys()):
            for i_name in protein_indicate:
                # Create plotting probability
                probability_predict = self.probability_data[value].loc[i_name].to_numpy()[0]

                # Create plotting DataFrame
                df = pd.DataFrame({'Feature_value': self.X_data.loc[i_name],
                                   'shapley_value': self.shapley_data_all[value].loc[i_name]})
                df['shapley_value_abs'] = df['shapley_value'].abs()
                df = df.sort_values(by='shapley_value_abs', ascending=False).iloc[:feature_remain]
                df['abs_rank'] = df['shapley_value_abs'].rank(ascending=True)
                
                feature_name = df.index.tolist()
                feature_value = df['Feature_value'].tolist()
                yticklabels = [f"{feature_name[i]} ({feature_value[i]:.4f})" for i in range(len(feature_name))]
                
                # Store positive and negative data in separate DataFrames
                positive_data = df[df['shapley_value'] > 0]
                negative_data = df[df['shapley_value'] <= 0]
                
                # Set PDF font parameters
                plt.rcParams['pdf.fonttype'] = 42

                # Create subplot with two horizontal histograms
                fig, ax = plt.subplots()

                # Plot horizontal histogram for positive data
                ax.barh(positive_data['abs_rank'].to_numpy(),
                        positive_data['shapley_value'], align='center', color=positive_color)

                # Plot horizontal histogram for negative data
                ax.barh(negative_data['abs_rank'].to_numpy(),
                        negative_data['shapley_value'], align='center', color=negative_color)
                        
                ax.set_yticks(df['abs_rank'].to_numpy())
                ax.set_yticklabels(yticklabels)

                # Add annotation for probability prediction value
                ax.text(1, 1, f"Based on {i_name} gene expression value, predicting {args.drug_name} {value} probability is: {probability_predict:.2f}",
                        transform=ax.transAxes,
                        verticalalignment='bottom', horizontalalignment='right')
                        
                ax.set_xlabel('IG Value')  # x-axis title
                ax.set_ylabel(f"Gene expression value of {i_name}")  # y-axis title

                fig.set_size_inches(plot_size)
                fig.savefig(save_path +
                            f"{file_name}_{value}_{i_name}_localplot.pdf",
                            dpi=1000, bbox_inches="tight")
                if png_plot:
                    fig.savefig(save_path +
                                f"{file_name}_{value}_{i_name}_localplot.png",
                                dpi=1000, bbox_inches="tight")
                plt.close(ax.figure)

# Main execution
# This section processes the SHAP analysis results. 
# Change the 'open_path' variable to process results from different methods (RF, DNN, DNN_IG)
open_path = args.open_path
save_path = args.save_path

# Check if path exists, create if it doesn't exist
if not os.path.exists(save_path):
    os.makedirs(save_path)

names = os.listdir(open_path)

# Read the target file based on the file keyword
df_train = process_data(args.key_name)
