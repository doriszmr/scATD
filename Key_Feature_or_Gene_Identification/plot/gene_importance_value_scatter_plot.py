
## Global parameter tuning
import os
import pandas as pd
import numpy as np
import re
from protloc_mex1.SHAP_plus import SHAP_importance_sum
import matplotlib.pyplot as plt
import argparse
parser = argparse.ArgumentParser(description="Set up and use command-line arguments for model handling.")


# Add arguments
parser.add_argument("--open_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/in', help="Path to open input data.")
parser.add_argument("--save_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/output/VAE_inference', help="Path to save output data.")

parser.add_argument("--key_name", type=str, default='feature')
parser.add_argument("--color", type=str, default='m')


# Parse the arguments
args = parser.parse_args()

def process_data(keyword):
    # Find file names containing a specific keyword and remove the '.csv' extension
    file_names = SHAP_importance_sum.str_contain(f'{keyword}_importance', names)
    file_names = [name.replace('.csv', '') for name in file_names]

    # Assume there's only one element in file_names
    file_name = file_names[0]

    # Create the full file path
    file_path = os.path.join(open_path, f'{file_name}.csv')

    # Read the CSV file using pandas
    df = pd.read_csv(file_path)

    # Set the 'cluster_name' column as the index and remove it after setting the index
    df.set_index('feature_name', inplace=True)

    # Add a row and calculate the sum of each row
    df.loc[:, 'feature importance'] = df.sum(axis=1)

    return df

def plot_histogram(df, title, bins_num=50):
    plt.figure(figsize=(5, 5))

    # Select data from the 'feature importance' row
    data = df.loc[:, 'feature importance']

    # Plot the histogram
    plt.hist(data, bins=bins_num, alpha=1, color=args.color)  # Set color to magenta
    plt.title(title)
    plt.xlabel('Total Gene Importance')
    plt.ylabel('Number of Genes')

    plt.rcParams['pdf.fonttype'] = 42

    # Save the image
    plt.savefig(f'{save_path}/{title}_histogram.png', dpi=1000)  # Set dpi to 1000 for png
    plt.savefig(f'{save_path}/{title}_histogram.pdf')  # Save as pdf, size 10x10 inches
    plt.close()  # Close the plot window

def plot_scatter_chart(df, title):
    plt.figure(figsize=(5, 5))

    # Sort the values of the 'feature importance' row and get the ranks
    data = df.loc[:, 'feature importance'].sort_values(ascending=False)
    ranks = np.arange(1, len(data) + 1)

    # Calculate log10-transformed values
    log_values = np.log10(data)

    # Plot the scatter chart
    plt.scatter(ranks, log_values, color=args.color,s=2)  # Set color to magenta
    plt.title(title)
    plt.xlabel('Gene Rank')
    plt.ylabel('log10(Gene Importance)')

    plt.rcParams['pdf.fonttype'] = 42

    # Save the image
    plt.savefig(f'{save_path}/{title}_scatter_chart.png', dpi=1000)  # Set dpi to 1000 for png
    plt.savefig(f'{save_path}/{title}_scatter_chart.pdf')  # Save as pdf, size 10x10 inches
    plt.close()  # Close the plot window


# This section processes the results of Random Forest (RF). If you need to process results for DNN or DNN_ig,
# simply change the 'open_path' variable to the corresponding directory.
open_path = args.open_path
save_path = args.save_path



# 检查路径是否存在
if not os.path.exists(save_path):
    # 如果路径不存在，则创建该路径
    os.makedirs(save_path)

names = os.listdir(open_path)

# Read the target file based on the file keyword
df_train = process_data(args.key_name)

# Plot histograms and scatter charts for df_train
plot_histogram(df_train, 'Gene importance rank')
plot_scatter_chart(df_train, 'Gene importance rank')

