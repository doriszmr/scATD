# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:47:16 2024

@author: qq102
"""





import os
import pandas as pd
import numpy as np
import argparse
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

def load_feature_importance(open_path, feature_name_col, sensitivity_col, resistance_col):
    """
    Load all .xlsx and .csv files from the specified directory and store them in a dictionary.
    The key is the substring before the first underscore in the filename.

    Parameters:
        open_path (str): Path to the input directory containing .xlsx and .csv files.
        feature_name_col (str): Column name for feature names.
        sensitivity_col (str): Column name for sensitivity values.
        resistance_col (str): Column name for resistance values.

    Returns:
        dict: Dictionary with keys as dataset identifiers and values as DataFrames.
    """
    feature_dict = {}
    filenames = os.listdir(open_path)
    supported_extensions = ['.xlsx', '.csv']

    for filename in filenames:
        full_path = os.path.join(open_path, filename)
        _, file_extension = os.path.splitext(filename)

        if file_extension.lower() in supported_extensions:
            # Extract key from filename (substring before first '_')
            key = filename.split('_')[0]
            try:
                if file_extension.lower() == '.xlsx':
                    df = pd.read_excel(full_path)
                elif file_extension.lower() == '.csv':
                    df = pd.read_csv(full_path)

                # Check if the required columns exist
                expected_columns = [feature_name_col, sensitivity_col, resistance_col]
                if not all(col in df.columns for col in expected_columns):
                    print(f"Warning: {filename} does not contain the required columns {expected_columns}. Skipping.")
                    continue

                # Ensure the sensitivity and resistance columns are numeric
                if not pd.api.types.is_numeric_dtype(df[sensitivity_col]) or not pd.api.types.is_numeric_dtype(df[resistance_col]):
                    print(f"Warning: Non-numeric data found in sensitivity or resistance columns in {filename}. Skipping.")
                    continue

                feature_dict[key] = df
                print(f"Loaded {file_extension.upper()} file: {filename} with key '{key}'")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Unsupported file format: {filename}. Skipping.")

    return feature_dict

def rank_features(feature_dict, feature_name_col, sensitivity_col, resistance_col):
    """
    For each DataFrame in the dictionary, add ranking columns for sensitivity, resistance, and their sum.

    Parameters:
        feature_dict (dict): Dictionary containing feature importance DataFrames.
        feature_name_col (str): Column name for feature names.
        sensitivity_col (str): Column name for sensitivity values.
        resistance_col (str): Column name for resistance values.

    Returns:
        dict: Dictionary with ranked DataFrames.
    """
    ranked_dict = {}
    for key, df in feature_dict.items():
        df = df.copy()
        # Calculate sum of sensitivity and resistance
        df['sensitivity_resistance_sum'] = df[sensitivity_col] + df[resistance_col]

        # Rank features: higher values get lower rank numbers (1 is highest)
        df['sensitivity_rank'] = df[sensitivity_col].rank(method='dense', ascending=False).astype(int)
        df['resistance_rank'] = df[resistance_col].rank(method='dense', ascending=False).astype(int)
        df['sum_rank'] = df['sensitivity_resistance_sum'].rank(method='dense', ascending=False).astype(int)

        ranked_dict[key] = df
        print(f"Ranked features for key '{key}'")
    return ranked_dict

def save_ranked_features(ranked_dict, save_path):
    """
    Save each ranked DataFrame to an .xlsx file in the specified save_path directory.

    Parameters:
        ranked_dict (dict): Dictionary containing ranked DataFrames.
        save_path (str): Path to the directory where ranked files will be saved.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")

    for key, df in ranked_dict.items():
        output_file = os.path.join(save_path, f"{key}_ranked.xlsx")
        try:
            df.to_excel(output_file, index=False)
            print(f"Saved ranked data to {output_file}")
        except Exception as e:
            print(f"Error saving {output_file}: {e}")
def query_feature_rank(ranked_dict, feature_names, feature_name_col):
    """
    Query the rankings of specific features across all datasets.
    Returns a DataFrame with the rankings.

    Parameters:
        ranked_dict (dict): Dictionary containing ranked DataFrames.
        feature_names (list): List of feature names to query.
        feature_name_col (str): Column name for feature names.

    Returns:
        pd.DataFrame: DataFrame containing the rankings of the specified features.
    """
    # Initialize a list to collect Series objects
    feature_series_list = []

    for feature_name in feature_names:
        feature_data = {}
        feature_present = False
        for key, df in ranked_dict.items():
            feature_row = df[df[feature_name_col] == feature_name]
            if not feature_row.empty:
                feature_present = True
                sensitivity_rank = feature_row['sensitivity_rank'].values[0]
                resistance_rank = feature_row['resistance_rank'].values[0]
                sum_rank = feature_row['sum_rank'].values[0]
                # Store rankings with dataset key as prefix
                feature_data[f"{key}_sensitivity_rank"] = sensitivity_rank
                feature_data[f"{key}_resistance_rank"] = resistance_rank
                feature_data[f"{key}_sum_rank"] = sum_rank
            else:
                print(f"Feature '{feature_name}' not found in dataset '{key}'.")
                # Assign NaN for missing data
                feature_data[f"{key}_sensitivity_rank"] = np.nan
                feature_data[f"{key}_resistance_rank"] = np.nan
                feature_data[f"{key}_sum_rank"] = np.nan
        if feature_present:
            # Create a Series for the feature and add to the list
            feature_series = pd.Series(feature_data, name=feature_name)
            feature_series_list.append(feature_series)
        else:
            print(f"Feature '{feature_name}' not found in any dataset.")

    if feature_series_list:
        # Concatenate all Series into a single DataFrame
        consolidated_results = pd.concat(feature_series_list, axis=1).T
        # Reset index to have feature names as a column
        consolidated_results.reset_index(inplace=True)
        consolidated_results.rename(columns={'index': feature_name_col}, inplace=True)
    else:
        # Return an empty DataFrame with the expected columns
        column_names = [feature_name_col]
        for key in ranked_dict.keys():
            column_names.extend([
                f"{key}_sensitivity_rank",
                f"{key}_resistance_rank",
                f"{key}_sum_rank"
            ])
        consolidated_results = pd.DataFrame(columns=column_names)

    return consolidated_results


def compute_and_plot_correlations(ranked_dict, save_path, feature_name_col, selected_genes):
    """
    计算每对数据集之间各排名类型的 Spearman 等级相关性，并生成相应的散点图。
    对于选定的基因，还生成单独的散点图。

    参数:
        ranked_dict (dict): 包含排名后 DataFrame 的字典。
        save_path (str): 保存图表的目录路径。
        feature_name_col (str): 特征名称的列名。
        selected_genes (list): 选定的基因名称列表。
    """

    # 初始化列表以存储相关性统计数据
    correlation_stats = []

    # 获取数据集的列表
    datasets = list(ranked_dict.keys())

    # 生成所有唯一的数据集对
    dataset_pairs = list(itertools.combinations(datasets, 2))

    for pair in dataset_pairs:
        dataset1, dataset2 = pair
        print(f"正在处理数据集对: {dataset1} vs {dataset2}")

        # 合并两个数据集，基于特征名称
        df1 = ranked_dict[dataset1]
        df2 = ranked_dict[dataset2]

        merged_df = pd.merge(df1, df2, on=feature_name_col, how='inner', suffixes=(f'_{dataset1}', f'_{dataset2}'))

        # 检查合并后的 DataFrame 是否为空
        if merged_df.empty:
            print(f"{dataset1} 和 {dataset2} 之间没有重叠的特征。跳过。")
            continue

        # 定义排名类型
        ranking_types = ['sensitivity_rank', 'resistance_rank', 'sum_rank']

        for ranking in ranking_types:
            rank1 = f"{ranking}_{dataset1}"
            rank2 = f"{ranking}_{dataset2}"

            # 检查排名列是否存在
            if rank1 not in merged_df.columns or rank2 not in merged_df.columns:
                print(f"排名列 {rank1} 或 {rank2} 不存在。跳过。")
                continue

            # 删除排名列中含有 NaN 的行
            plot_df = merged_df[[rank1, rank2, feature_name_col]].dropna()

            if plot_df.empty:
                print(f"排名类型 '{ranking}' 在 {dataset1} 和 {dataset2} 之间没有可用的数据。跳过。")
                continue

            # 计算 Spearman 等级相关系数
            rho, pval = spearmanr(plot_df[rank1], plot_df[rank2])

            # 将统计结果添加到列表
            correlation_stats.append({
                'Dataset 1': dataset1,
                'Dataset 2': dataset2,
                'Ranking Type': ranking,
                'Correlation Rho': rho,
                'P-Value': pval,
                'Gene': 'All Genes'
            })

            # =======================
            # 生成散点图
            # =======================

            # 创建散点图
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=plot_df[rank1], y=plot_df[rank2], alpha=0.6)

            # 绘制 y=x 直线
            min_rank = min(plot_df[rank1].min(), plot_df[rank2].min())
            max_rank = max(plot_df[rank1].max(), plot_df[rank2].max())
            plt.plot([min_rank, max_rank], [min_rank, max_rank], 'r--', label='y=x')

            # 设置坐标轴标签和标题
            plt.xlabel(f"{dataset1} {ranking}")
            plt.ylabel(f"{dataset2} {ranking}")
            plt.title(f"Spearman Correlation: {dataset1} vs {dataset2} - {ranking}\nρ={rho:.2f}, p={pval:.3f}")
            plt.legend()

            # 保存图表为PNG和PDF
            plot_filename_png = f"Spearman_{ranking}_{dataset1}_vs_{dataset2}.png"
            plot_filepath_png = os.path.join(save_path, plot_filename_png)
            plt.tight_layout()
            plt.savefig(plot_filepath_png, dpi=1000)
            plt.close()
            print(f"已保存图表: {plot_filepath_png}")

            # 保存为PDF，DPI设置为1000
            plot_filename_pdf = f"Spearman_{ranking}_{dataset1}_vs_{dataset2}.pdf"
            plot_filepath_pdf = os.path.join(save_path, plot_filename_pdf)
            plt.figure(figsize=(8, 6))
            sns.scatterplot(x=plot_df[rank1], y=plot_df[rank2], alpha=0.6)
            plt.plot([min_rank, max_rank], [min_rank, max_rank], 'r--', label='y=x')
            plt.xlabel(f"{dataset1} {ranking}")
            plt.ylabel(f"{dataset2} {ranking}")
            plt.title(f"Spearman Correlation: {dataset1} vs {dataset2} - {ranking}\nρ={rho:.2f}, p={pval:.3f}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_filepath_pdf, dpi=1000)
            plt.close()
            print(f"已保存图表: {plot_filepath_pdf}")

            # =======================
            # 生成 Hexbin 图
            # =======================
            plt.figure(figsize=(8, 6))
            hb = plt.hexbin(plot_df[rank1], plot_df[rank2], gridsize=50, cmap='Blues', mincnt=1, alpha=0.6)
            plt.colorbar(hb, label='Count')

            # 绘制 y=x 直线
            plt.plot([min_rank, max_rank], [min_rank, max_rank], 'r--', label='y=x')

            # 设置坐标轴标签和标题
            plt.xlabel(f"{dataset1} {ranking}")
            plt.ylabel(f"{dataset2} {ranking}")
            plt.title(f"Spearman Correlation (Hexbin): {dataset1} vs {dataset2} - {ranking}\nρ={rho:.2f}, p={pval:.3f}")
            plt.legend()

            # 保存 Hexbin 图为PNG和PDF
            hexbin_filename_png = f"Spearman_Hexbin_{ranking}_{dataset1}_vs_{dataset2}.png"
            hexbin_filepath_png = os.path.join(save_path, hexbin_filename_png)
            plt.tight_layout()
            plt.savefig(hexbin_filepath_png, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"已保存 Hexbin 图: {hexbin_filepath_png}")

            hexbin_filename_pdf = f"Spearman_Hexbin_{ranking}_{dataset1}_vs_{dataset2}.pdf"
            hexbin_filepath_pdf = os.path.join(save_path, hexbin_filename_pdf)
            plt.figure(figsize=(8, 6))
            hb = plt.hexbin(plot_df[rank1], plot_df[rank2], gridsize=50, cmap='Blues', mincnt=1, alpha=0.8)
            plt.colorbar(hb, label='Count')
            plt.plot([min_rank, max_rank], [min_rank, max_rank], 'r--', label='y=x')
            plt.xlabel(f"{dataset1} {ranking}")
            plt.ylabel(f"{dataset2} {ranking}")
            plt.title(f"Spearman Correlation (Hexbin): {dataset1} vs {dataset2} - {ranking}\nρ={rho:.2f}, p={pval:.3f}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(hexbin_filepath_pdf, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"已保存 Hexbin 图: {hexbin_filepath_pdf}")


            # 如果有选定的基因，生成单独的散点图
            if selected_genes:
                selected_plot_df = plot_df[plot_df[feature_name_col].isin(selected_genes)]
                if not selected_plot_df.empty:
                    # 计算选定基因的 Spearman 等级相关系数
                    if len(selected_plot_df) >= 2:
                        rho_selected, pval_selected = spearmanr(selected_plot_df[rank1], selected_plot_df[rank2])
                    else:
                        rho_selected, pval_selected = np.nan, np.nan
                        print(f"选定基因数少于2个，无法计算 Spearman 相关系数。")

                    plt.figure(figsize=(8, 6))
                    sns.scatterplot(x=selected_plot_df[rank1], y=selected_plot_df[rank2],
                                    hue=selected_plot_df[feature_name_col], s=100, edgecolor='k')
                    # 绘制 y=x 直线
                    plt.plot([min_rank, max_rank], [min_rank, max_rank], 'r--', label='y=x')
                    # 设置坐标轴标签和标题
                    plt.xlabel(f"{dataset1} {ranking}")
                    plt.ylabel(f"{dataset2} {ranking}")
                    if not np.isnan(rho_selected):
                        plt.title(
                            f"Selected Genes Spearman Correlation: {dataset1} vs {dataset2} - {ranking}\nρ={rho_selected:.2f}, p={pval_selected:.3f}")
                    else:
                        plt.title(
                            f"Selected Genes Spearman Correlation: {dataset1} vs {dataset2} - {ranking}\n无法计算相关性（样本数不足）")
                    plt.legend(title='Selected Genes')
                    plt.tight_layout()
                    # 保存选定基因的图表为PDF，DPI设置为1000
                    selected_plot_filename_pdf = f"Spearman_{ranking}_{dataset1}_vs_{dataset2}_selected_genes.pdf"
                    selected_plot_filepath_pdf = os.path.join(save_path, selected_plot_filename_pdf)
                    plt.savefig(selected_plot_filepath_pdf, dpi=1000)
                    plt.close()
                    print(f"已保存选定基因的图表: {selected_plot_filepath_pdf}")
                else:
                    print(f"在 {dataset1} 和 {dataset2} 的排名类型 '{ranking}' 中，没有选定基因的数据。")
    # 将收集到的相关性统计数据转换为 DataFrame
    stats_df = pd.DataFrame(correlation_stats)

    # 定义 Excel 文件的完整路径
    excel_path = os.path.join(save_path, 'ALL_Spearman_Correlations.xlsx')


    stats_df.to_excel(excel_path, index=False)
    print(f"所有相关性统计结果已保存到 Excel 文件: {excel_path}")

    return stats_df  # 返回统计数据 DataFrame

def plot_correlation_heatmaps(stats_df, save_path,  dataset_order=None, excel_filename="correlation_results.xlsx"):
    """
    基于 stats_df 生成 Spearman 相关性热图，并保存为 PNG 文件。

    参数:
        stats_df (pd.DataFrame): 包含相关性统计结果的 DataFrame。
        save_path (str): 保存热图的目录路径。
        excel_filename (str): 用于命名热图文件的一部分（可选）。
    """
    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    # 仅选择 'All Genes' 的相关性，避免混淆 'Selected Genes' 数据
    all_genes_df = stats_df[stats_df['Gene'] == 'All Genes']

    # 获取所有排名类型
    ranking_types = all_genes_df['Ranking Type'].unique()

    for ranking in ranking_types:
        # 筛选当前排名类型的数据
        ranking_df = all_genes_df[all_genes_df['Ranking Type'] == ranking]

        # 创建一个透视表，以便绘制热图
        pivot_table = ranking_df.pivot(index='Dataset 1', columns='Dataset 2', values='Correlation Rho')

        # 获取所有数据集的名称
        all_datasets = sorted(set(ranking_df['Dataset 1']).union(ranking_df['Dataset 2']))

        # 如果指定了 dataset_order，则验证其完整性并应用排序
        if dataset_order:
            # 检查 dataset_order 中的所有数据集是否存在于 all_datasets 中
            missing_datasets = set(dataset_order) - set(all_datasets)
            if missing_datasets:
                raise ValueError(f"以下数据集在 stats_df 中未找到: {missing_datasets}")

            # 使用指定的排序
            ordered_datasets = dataset_order
        else:
            # 使用默认排序
            ordered_datasets = all_datasets

        # # 由于数据集对是无序的（Dataset 1 < Dataset 2），为了对称热图，我们需要填充对称位置
        # pivot_full = pivot_table.copy()
        # 重新索引 pivot_table 以应用排序
        pivot_table = pivot_table.reindex(index=ordered_datasets, columns=ordered_datasets)

        # 填充对角线为1（自相关）
        np.fill_diagonal(pivot_table.values, 1)

        # 绘制热图
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1,
                    linewidths=.5, square=True, cbar_kws={"shrink": .5})
        plt.title(f"Spearman Correlation Heatmap - {ranking} (All Genes)")
        plt.tight_layout()

        # 保存热图为 PNG
        heatmap_filename_png = f"Spearman_Correlation_Heatmap_{ranking}.png"
        heatmap_filepath_png = os.path.join(save_path, heatmap_filename_png)
        plt.savefig(heatmap_filepath_png, dpi=300)
        print(f"已保存热图: {heatmap_filepath_png}")

        # 保存热图为 PDF
        heatmap_filename_pdf = f"Spearman_Correlation_Heatmap_{ranking}.pdf"
        heatmap_filepath_pdf = os.path.join(save_path, heatmap_filename_pdf)
        plt.savefig(heatmap_filepath_pdf, dpi=1000)
        print(f"已保存热图: {heatmap_filepath_pdf}")
        plt.close()




def main():
    parser = argparse.ArgumentParser(description="Process feature importance files.")
    parser.add_argument('--open_path', type=str, required=True, help='Path to the input directory containing .xlsx files.')
    parser.add_argument('--save_path', type=str, required=True, help='Path to the directory where ranked files will be saved.')

    # Updated to accept multiple feature names
    parser.add_argument(
        '--query_feature',
        type=str,
        nargs='+',  # Allows multiple values
        default=["A1BG", "TP53"],  # Default includes both values
        help='Optional: Feature names to query rankings. Provide multiple feature names separated by space.'
    )

    # Arguments for expected column names
    parser.add_argument('--feature_name_col', type=str, default='feature_name', help='Column name for feature names. Default is "feature_name".')
    parser.add_argument('--sensitivity_col', type=str, default='sensitivity', help='Column name for sensitivity values. Default is "sensitivity".')
    parser.add_argument('--resistance_col', type=str, default='resistance', help='Column name for resistance values. Default is "resistance".')

    parser.add_argument(
        '--indicate_data_name_seq',
        type=str,
        nargs='+',  # Allows multiple values
        default=None,  # Default includes both values
    )


    args = parser.parse_args()
    # 设置PDF字体参数
    plt.rcParams['pdf.fonttype'] = 42  # 确保嵌入字体而不是将其转换为路径

    # Load feature importance files
    feature_dict = load_feature_importance(
        args.open_path,
        args.feature_name_col,
        args.sensitivity_col,
        args.resistance_col
    )

    if not feature_dict:
        print("No valid .xlsx files found. Exiting.")
        return

    # Rank features
    ranked_dict = rank_features(
        feature_dict,
        args.feature_name_col,
        args.sensitivity_col,
        args.resistance_col
    )

    # Save ranked features
    save_ranked_features(ranked_dict, args.save_path)

    # If query features are specified, perform the query
    if args.query_feature:
        query_df = query_feature_rank(
            ranked_dict,
            args.query_feature,
            args.feature_name_col
        )
        if not query_df.empty:
            print(f"\nRankings for queried features:")
            print(query_df)
            # Save the consolidated query results
            query_output_file = os.path.join(args.save_path, f"queried_features_rankings.xlsx")
            try:
                # Save all results in one sheet
                query_df.to_excel(query_output_file, index=False)
                print(f"Saved query results to {query_output_file}")
            except Exception as e:
                print(f"Error saving query results: {e}")
        else:
            print(f"No rankings found for the specified features.")

    # 计算 Spearman 相关性并生成图表
    stats_df = compute_and_plot_correlations(
        ranked_dict,
        args.save_path,
        args.feature_name_col,
        args.query_feature  # 使用 query_feature 作为 selected_genes
    )

    # 调用热图生成函数
    plot_correlation_heatmaps(stats_df, args.save_path, args.indicate_data_name_seq )
if __name__ == "__main__":
    main()

