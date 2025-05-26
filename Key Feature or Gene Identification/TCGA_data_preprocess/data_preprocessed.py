# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:44:41 2024

@author: qq102
"""





import os

import torch
import sys

import re

import argparse

import numpy as np
import random
import scanpy as sc
import pandas as pd
parser = argparse.ArgumentParser(description="Set up and use command-line arguments for model handling.")



def strict_str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Only "True" or "False" are allowed.')


# Add arguments
parser.add_argument("--open_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/in', help="Path to open input data.")
parser.add_argument("--save_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/output/VAE_inference', help="Path to save output data.")

parser.add_argument("--file_prefix", type=str, default='scRNA-seq_panglao_0_1_', help="Prefix for files to handle.")


parser.add_argument('--gene_ID', type=str, default='Ensembl_ID', help='label of classes.')
parser.add_argument('--seed_set', type=int, default=42, help='Number of random seed.')


parser.add_argument('--feature_name_to_gene', type=str,help= 'file path of scfoundation_19264_gene_index.xlsx')

parser.add_argument("--gene_shift", type=strict_str2bool, default=False, help="if need ENSGxxxx convert to gene name")

parser.add_argument('--gene_ENG_mapping_file', type=str,default= 'mart_export.txt',help= 'file path of ENSGxxxx convert to gene name')





# Parse the arguments
args = parser.parse_args()



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(args.seed_set)





# 检查目录是否存在，如果不存在则创建
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)







def load_gene_symbol_to_ensembl_mapping(mapping_file):
    """
    Load gene symbol to Ensembl ID mapping from a text file.

    Parameters:
    - mapping_file: Path to the mapping file (e.g., a text file).

    Returns:
    - A dictionary mapping gene symbols to Ensembl IDs.
    """
    # Read the text file assuming it's delimited by commas
    mapping_df = pd.read_csv(mapping_file, sep=',')
    # Create a dictionary mapping 'Gene name' to 'Gene stable ID'
    gene_symbol_to_ensembl = dict(zip(mapping_df['Gene name'], mapping_df['Gene stable ID']))
    return gene_symbol_to_ensembl


##基因转换函数

def convert_ensembl_to_gene_names_df_vectorized(df, ensembl_to_gene_symbol, none_name_save, match_name_save, gene_ID):
    """
    使用向量化操作将 DataFrame 中的 Ensembl ID（位于指定的 gene_ID 列）转换为基因符号（Gene Names），并将其存储在新的列中。

    参数:
    - df: 一个 pandas DataFrame，包含一个指定的基因列（例如 'gene_ID'），其余列为细胞名。
    - ensembl_to_gene_symbol: 一个字典，键为 Ensembl ID，值为基因符号。
    - none_name_save: 无法映射的 Ensembl ID 将保存到的 Excel 文件路径。
    - match_name_save: 成功映射的 Ensembl ID 及其基因符号将保存到的 Excel 文件路径。
    - gene_ID: 字符串，表示 df 中用于存储 Ensembl ID 的列名（例如 'gene_ID'）。

    返回:
    - modified_df: 修改后的 DataFrame，添加了 'Gene_Name' 列。
    - modified_gene_names: 修改后的基因名称列表（Gene Names）。
    - gene_names_series: Gene Names 的 Series。
    """
    # 检查指定的 gene_ID 列是否存在
    if gene_ID not in df.columns:
        raise ValueError(f"输入的 DataFrame 必须包含名为 '{gene_ID}' 的列作为基因名称。")

    # 打印原始的基因数量
    print(f"原始基因数量: {df.shape[0]}")

    # 打印前5个基因名称以判断基因名称的格式
    print("前5个基因名称为:", df[gene_ID].head(5).tolist())


    # 处理 Ensembl ID，提取下划线和小数点前的主要部分
    df['Primary_Ensembl_ID'] = df[gene_ID].astype(str).apply(
        lambda x: x.split('_')[0].split('.')[0] if x.startswith('ENS') else x
    )

    # 映射 Ensembl ID 到基因符号
    df['Gene_Name'] = df['Primary_Ensembl_ID'].map(ensembl_to_gene_symbol)

    # 保存无法映射的 Ensembl ID 到 Excel
    none_entries = df[df['Gene_Name'].isnull()]
    # none_entries.to_excel(none_name_save, index=False)
    # print(f"无法映射的 Ensembl ID 已保存到: {none_name_save}")

    # 记录无法映射的基因数量
    num_none_genes = none_entries.shape[0]
    print(f"无法映射的基因数量: {num_none_genes}")

    # 过滤掉无法映射的基因
    modified_df = df[df['Gene_Name'].notnull()].copy()

    # 验证过滤的基因数量是否一致
    num_genes_after_filter = modified_df.shape[0]
    num_genes_before_filter = df.shape[0]
    num_filtered_out = num_genes_before_filter - num_genes_after_filter
    print(f"原始基因数量: {num_genes_before_filter}, 过滤后的基因数量: {num_genes_after_filter}")
    print(f"被过滤的基因数量: {num_filtered_out}，与无法映射的基因数量相等: {num_filtered_out == num_none_genes}")

    # 将映射成功的基因保存到 Excel
    combined_df = modified_df[[gene_ID, 'Gene_Name']].copy()
    combined_df.to_excel(match_name_save, index=False)
    print(f"映射成功的基因已保存到: {match_name_save}")

    # 获取修改后的基因名称列表和基因符号
    modified_gene_names = modified_df['Gene_Name'].tolist()
    gene_names_series = modified_df['Gene_Name']

    return modified_df, modified_gene_names, gene_names_series


##基因名筛选和匹配函数
def main_gene_selection(X_df, gene_list):
    # # 处理重复列名，保留第一个出现的列
    # X_df = X_df.loc[:, ~X_df.columns.duplicated()]

    # 或者，对重复列进行聚合（选择适合您的方法）
    X_df = X_df.groupby(X_df.columns, axis=1).mean()

    to_fill_columns = list(set(gene_list) - set(X_df.columns))

    unmatched_genes = [gene for gene in gene_list if gene not in X_df.columns]

    # Identify genes that are in X_df.columns but not in gene_list
    extra_genes = [gene for gene in X_df.columns if gene not in gene_list]

    # Count the number of unmatched genes
    unmatched_genes_count = len(unmatched_genes)
    extra_genes_count = len(extra_genes)
    if unmatched_genes_count > 0:
        print(f"Number of unmatched genes: {unmatched_genes_count}")
        # print(f"Unmatched genes: {unmatched_genes}")
    else:
        print("All genes matched.")

    if extra_genes_count > 0:
        print(f"Number of extra genes in X_df: {extra_genes_count}")
        # print(f"Extra genes: {extra_genes}")
    else:
        print("No extra genes in X_df.")

    padding_df = pd.DataFrame(np.zeros((X_df.shape[0], len(to_fill_columns))),
                              columns=to_fill_columns,
                              index=X_df.index)
    X_df = pd.DataFrame(np.concatenate([df.values for df in [X_df, padding_df]], axis=1),
                        index=X_df.index,
                        columns=list(X_df.columns) + list(padding_df.columns))

    ##统一基因顺序
    X_df = X_df[gene_list]

    var = pd.DataFrame(index=X_df.columns)
    var['mask'] = [1 if i in to_fill_columns else 0 for i in list(var.index)]
    return X_df, to_fill_columns, var, unmatched_genes, extra_genes


##读入生存数据和病人的表达量
# 文件路径

open_path = args.open_path
save_path =args.save_path
# 获取指定路径下的所有文件
all_files = os.listdir(open_path)

# 初始化文件路径变量
tpm_file = None
survival_file = None

# 正则表达式匹配 tpm 和 survival 文件
for filename in all_files:
    if re.search(r'tpm\.tsv$', filename):
        tpm_file = os.path.join(open_path, filename)
    elif re.search(r'survival\.tsv$', filename):
        survival_file = os.path.join(open_path, filename)

# 读取 tpm 文件
if tpm_file:
    tpm_data = pd.read_csv(tpm_file, sep='\t')
    print("TPM数据预览：")
    print(tpm_data.head())
else:
    print("未找到TPM数据文件")

# 读取 survival 文件
if survival_file:
    survival_data = pd.read_csv(survival_file, sep='\t')
    print("生存数据预览：")
    print(survival_data.head())
else:
    print("未找到生存数据文件")




# 使用提供的映射文件路径加载基因映射

if args.gene_shift:
    mapping_file  = args.gene_ENG_mapping_file
    mapping_path = os.path.join(open_path, mapping_file)
    gene_symbol_to_ensembl = load_gene_symbol_to_ensembl_mapping(mapping_path)

    # 假设 gene_symbol_to_ensembl 是基因符号到 Ensembl ID 的字典
    ensembl_to_gene_symbol = {v: k for k, v in gene_symbol_to_ensembl.items()}


    # 调用转换函数
    # 调用转换函数，使用反向映射字典
    # 调用向量化转换函数



    modified_df, modified_gene_names, gene_names_series = convert_ensembl_to_gene_names_df_vectorized(
        df=tpm_data,
        ensembl_to_gene_symbol=ensembl_to_gene_symbol,
        none_name_save=os.path.join(save_path,f'{args.file_prefix}_unmapped_ensembl_ids.xlsx'),
        match_name_save=os.path.join(save_path,f'{args.file_prefix}_mapped_genes.xlsx'),
        gene_ID=args.gene_ID
    )
    tpm_data = modified_df
    tpm_data.drop(columns=['Primary_Ensembl_ID'], inplace=True)



# 使用绝对路径读入 Excel 文件,scfoundation_19264_gene_name

file_path = args.feature_name_to_gene

data_feature_to_gene = pd.read_excel(file_path)


# 移除 'Ensembl_ID' 列
tpm_data.drop(columns=[args.gene_ID], inplace=True)




# 设置 'Gene_name' 列为索引
tpm_data.set_index('Gene_Name', inplace=True)

# 转置数据框
transposed_df = tpm_data.transpose()


gene_list = list(data_feature_to_gene['gene_name'])

X_df, to_fill_columns, var,_,_ = main_gene_selection(transposed_df, gene_list)

adata = sc.AnnData(X_df)

# 保存为 .h5ad 文件
output_path = os.path.join(save_path,f'{args.file_prefix}_19264_preprocessed.h5ad')
adata.write(output_path)

print(f"DataFrame has been saved to {output_path}")