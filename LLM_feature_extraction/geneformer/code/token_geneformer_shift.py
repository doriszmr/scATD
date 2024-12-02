import argparse
import scanpy as sc
import os
import numpy as np
import pandas as pd




def strict_str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Only "True" or "False" are allowed.')



# 创建命令行参数解析器
parser = argparse.ArgumentParser(description="Set up and use command-line arguments for model handling.")

# 添加参数
parser.add_argument("--open_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/in/your_data.h5ad', help="Path to open input data.")
parser.add_argument("--save_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/output/VAE_inference', help="Path to save output data.")
parser.add_argument("--open_path_conference_data", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/conference_data', help="Path to open conference data.")
parser.add_argument("--mapping_file", type=str, help="Filename of the gene mapping file to load.")
parser.add_argument("--G38_convert", type=strict_str2bool, default=False, help="if G38 need, defult only result in G37.")

# 解析命令行参数
args = parser.parse_args()

# 将参数赋值给变量
open_path = args.open_path
save_path = args.save_path

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

# 使用提供的映射文件路径加载基因映射
mapping_path = os.path.join(args.open_path_conference_data, args.mapping_file)
gene_symbol_to_ensembl = load_gene_symbol_to_ensembl_mapping(mapping_path)



# 使用 os.path 来获取文件名（不含后缀）
file_name = os.listdir(open_path)[0] # 获取文件名不含后缀

input_file_name = os.path.splitext(file_name)[0]


output_file_path = os.path.join(save_path, f"{input_file_name}_n_counts.h5ad")  # 生成输出文件路径

# 使用Scanpy加载数据
adata = sc.read_h5ad(os.path.join(open_path,file_name))

# 计算每个细胞的总计数，并将其存储在 'n_counts' 列中
if isinstance(adata.X, np.ndarray):
    adata.obs["n_counts"] = np.sum(adata.X, axis=1)
else:
    # 如果是稀疏矩阵，使用 sum 方法保持稀疏性
    adata.obs["n_counts"] = np.array(adata.X.sum(axis=1)).flatten()

# 输出计算结果
print(adata.obs["n_counts"])

def convert_gene_names_to_ensembl(adata, gene_symbol_to_ensembl, none_name_save, match_name_save):
    """
    将 adata.var_names 中的基因名称转换为 Ensembl ID，并存储在 adata.var['ensembl_id'] 中。

    参数：
    - adata: An AnnData 对象。
    - gene_symbol_to_ensembl: 一个字典，将基因符号映射到 Ensembl ID。
    - none_name_save: 保存无法映射的基因的文件名。
    - match_name_save: 保存成功映射的基因的文件名。

    返回：
    - adata: 更新后的 AnnData 对象。
    - modified_gene_names: 修改后的基因名称列表。
    - ENSG_name: Ensembl ID 列表。
    """
    # 打印原始的维度
    print(f"原始adata的维度: {adata.shape}")

    # 打印前5行基因名称以判断基因名称的格式
    print("前5行基因名称为:", adata.var_names[:5])

    ensembl_ids = []
    for gene_name in adata.var_names:
        # 如果基因名称是 'ENSxxxxx_GeneSymbol' 格式，提取 Ensembl ID
        if '_' in gene_name and gene_name.startswith('ENS'):
            ensembl_id = gene_name.split('_')[0]  # 提取下划线前的 Ensembl ID 部分
            ensembl_ids.append(ensembl_id)

        # 如果基因名称是 Ensembl ID
        elif gene_name.startswith('ENS'):
            ensembl_ids.append(gene_name)  # 如果已经是 Ensembl ID，直接添加

        # 如果基因名称是基因符号，尝试通过映射字典获取 Ensembl ID
        else:
            ensembl_id = gene_symbol_to_ensembl.get(gene_name)  # 从字典中获取映射的 Ensembl ID
            if ensembl_id:
                ensembl_ids.append(ensembl_id)
            else:
                # 如果无法映射，标记为 None
                ensembl_ids.append(None)

    # 将映射的 Ensembl ID 添加到 adata.var 中
    adata.var['ensembl_id'] = ensembl_ids

    # 保存无法映射的基因符号
    none_mask = adata.var['ensembl_id'].isnull()
    none_entries = adata.var[none_mask]
    none_entries.to_excel(none_name_save, index=True)
    print(f"无法映射的基因已保存到: {none_name_save}")

    # 记录无法映射的基因数量
    num_none_genes = none_entries.shape[0]
    print(f"无法映射的基因数量: {num_none_genes}")

    # 过滤掉无法映射的基因
    adata = adata[:, ~none_mask.values]

    # 验证过滤的基因数量是否一致
    num_genes_after_filter = adata.shape[1]
    num_genes_before_filter = len(ensembl_ids)
    num_filtered_out = num_genes_before_filter - num_genes_after_filter
    print(f"原始基因数量: {num_genes_before_filter}, 过滤后的基因数量: {num_genes_after_filter}")
    print(f"被过滤的基因数量: {num_filtered_out}，与无法映射的基因数量相等: {num_filtered_out == num_none_genes}")

    # 获取修改后的基因名称和 Ensembl ID
    modified_gene_names = adata.var_names.tolist()
    ENSG_name = adata.var['ensembl_id']

    # 将修改后的基因名称和 Ensembl ID 合并为一个 DataFrame
    combined_df = pd.DataFrame({
        'Gene_Name': modified_gene_names,
        'Ensembl_ID': ENSG_name
    })

    # 将合并后的 DataFrame 保存为 Excel 文件
    combined_df.to_excel(match_name_save, index=False)
    print(f"成功映射的基因已保存到: {match_name_save}")

    return adata, modified_gene_names, ENSG_name





none_name_save = os.path.join(save_path, f"{input_file_name}_none_ensembl_ids.xlsx")

match_name_save = os.path.join(save_path, f"{input_file_name}_match_ensembl_ids.xlsx")


adata_convert, modified_gene_names,ENSG_name = convert_gene_names_to_ensembl(adata, gene_symbol_to_ensembl, none_name_save,match_name_save)

# 保存修改后的基因名称到文件（可选）
modified_genes_file = os.path.join(save_path, f"{input_file_name}_modified_gene_names.txt")
with open(modified_genes_file, 'w') as f:
    for gene_name in modified_gene_names:
        f.write(f"{gene_name}\n")
print(f"修改后的基因名称已保存至: {modified_genes_file}")

# 保存修改后的基因名称到文件（可选）
ENSG_name_file = os.path.join(save_path, f"{input_file_name}_ENSG_name_gene_names.txt")
with open(ENSG_name_file, 'w') as f:
    for gene_name in ENSG_name:
        f.write(f"{gene_name}\n")
print(f"修改后的基因名称已保存至: {ENSG_name_file}")



# 保存结果到指定路径
adata_convert.write(output_file_path)

print(f"数据已保存至: {output_file_path}")



#
# if args.G38_convert:
#
#     from pybiomart import Dataset
#
#     def convert_gene_names_to_ensembl_grch38(adata):
#         """
#         Convert gene names in adata.var_names to Ensembl IDs using BioMart (GRCh38).
#
#         Parameters:
#         - adata: An AnnData object.
#         """
#         # Connect to Ensembl GRCh38 dataset
#         dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
#         # Query BioMart to convert gene symbols to Ensembl IDs
#
#         # # Print all available filters and attributes for verification
#         # print("Available filters:", list(dataset.filters.keys()))
#         # print("Available attributes:", list(dataset.attributes.keys()))
#
#         # Use 'hgnc_symbol' as the filter to query by gene symbols
#         results = dataset.query(attributes=['ensembl_gene_id'],
#                                 filters={'gene_id': adata.var_names.tolist()})
#
#         # Create a mapping from gene symbols to Ensembl IDs
#         gene_symbol_to_ensembl = dict(zip(results['gene_id'], results['ensembl_gene_id']))
#
#
#         ensembl_ids = []
#         for gene_name in adata.var_names:
#             ensembl_id = gene_symbol_to_ensembl.get(gene_name)
#             if ensembl_id:
#                 ensembl_ids.append(ensembl_id)
#             else:
#                 ensembl_ids.append(None)
#         adata.var['ensembl_id'] = ensembl_ids
#
#         # Save rows with None Ensembl IDs to Excel
#         none_mask = adata.var['ensembl_id'].isnull()
#         none_entries = adata.var[none_mask]
#         none_name_save = os.path.join(save_path, "g38_none_ensembl_ids.xlsx")
#         none_entries.to_excel(none_name_save, index=True)
#         print(f"Entries with None Ensembl IDs for GRCh38 have been saved to: {none_name_save}")
#
#         # Filter out genes that couldn't be mapped
#         adata = adata[:, [eid is not None for eid in ensembl_ids]]
#         # 返回修改后的 AnnData 对象
#         modified_gene_names = adata.var_names.tolist()  # 获取修改后的基因名称列表
#         return adata,modified_gene_names
#
#
#     # GRCh38转换
#     data_grch38,modified_gene_names_grch38 = convert_gene_names_to_ensembl_grch38(adata.copy())
#
#     # 保存修改后的基因名称到文件（可选）
#     modified_gene_names_grch38 = os.path.join(save_path, f"{input_file_name}_grch38_modified_gene_names.txt")
#     with open(modified_gene_names_grch38, 'w') as f:
#         for gene_name in modified_gene_names_grch38:
#             f.write(f"{gene_name}\n")
#     print(f"修改后的基因名称已保存至: {modified_gene_names_grch38}")
#
#     output_file_path_grch38 = os.path.join(save_path, "_grch38_n_counts.h5ad")
#     data_grch38.write(output_file_path_grch38)
#     print(f"GRCh38 数据已保存至: {output_file_path_grch38}")






#
#
# import argparse
# import scanpy as sc
# import os
# import numpy as np
# import pandas as pd
#
# # 创建命令行参数解析器
# parser = argparse.ArgumentParser(description="Set up and use command-line arguments for model handling.")
#
# # 添加参数
# parser.add_argument("--open_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/in/your_data.h5ad',
#                     help="Path to open input data.")
# parser.add_argument("--save_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/output/VAE_inference',
#                     help="Path to save output data.")
# parser.add_argument("--open_path_conference_data", type=str,
#                     default='/home/luozeyu/desktop/VAE_pretraining/data/conference_data',
#                     help="Path to open conference data.")
# parser.add_argument("--mapping_file", type=str, help="Filename of the gene mapping file to load.")
#
# # 解析命令行参数
# args = parser.parse_args()
#
# # 将参数赋值给变量
# open_path = args.open_path
# save_path = args.save_path
#
#
# def load_gene_symbol_to_ensembl_mapping(mapping_file):
#     """
#     Load gene symbol to Ensembl ID mapping from a text file.
#
#     Parameters:
#     - mapping_file: Path to the mapping file (e.g., a text file).
#
#     Returns:
#     - A dictionary mapping gene symbols to Ensembl IDs.
#     """
#     # Read the text file assuming it's delimited by commas
#     mapping_df = pd.read_csv(mapping_file, sep=',')
#     # Create a dictionary mapping 'Gene name' to 'Gene stable ID'
#     gene_symbol_to_ensembl = dict(zip(mapping_df['Gene name'], mapping_df['Gene stable ID']))
#     return gene_symbol_to_ensembl
#
#
# def convert_gene_names_to_ensembl_grch37(adata, gene_symbol_to_ensembl, none_name_save):
#     """
#     Convert gene names in adata.var_names to Ensembl IDs using a provided mapping file (GRCh37).
#
#     Parameters:
#     - adata: An AnnData object.
#     - gene_symbol_to_ensembl: A dictionary mapping gene symbols to Ensembl IDs.
#     """
#     ensembl_ids = []
#     for gene_name in adata.var_names:
#         ensembl_id = gene_symbol_to_ensembl.get(gene_name)
#         if ensembl_id:
#             ensembl_ids.append(ensembl_id)
#         else:
#             # Mark as None if mapping fails
#             ensembl_ids.append(None)
#     adata.var['ensembl_id_grch37'] = ensembl_ids
#
#     # Save rows with None Ensembl IDs to Excel
#     none_mask = adata.var['ensembl_id_grch37'].isnull()
#     none_entries = adata.var[none_mask]
#     none_entries.to_excel(none_name_save, index=True)
#
#     # Filter out genes that couldn't be mapped
#     adata = adata[:, [eid is not None for eid in ensembl_ids]]
#     return adata
#
#
# def convert_gene_names_to_ensembl_grch38(adata):
#     """
#     Convert gene names in adata.var_names to Ensembl IDs using BioMart (GRCh38).
#
#     Parameters:
#     - adata: An AnnData object.
#     """
#     # Connect to Ensembl GRCh38 dataset
#     dataset = Dataset(name='hsapiens_gene_ensembl', host='http://www.ensembl.org')
#     # Query BioMart to convert gene symbols to Ensembl IDs
#     results = dataset.query(attributes=['hgnc_symbol', 'ensembl_gene_id'],
#                             filters={'hgnc_symbol': adata.var_names.tolist()})
#     gene_symbol_to_ensembl = dict(zip(results['hgnc_symbol'], results['ensembl_gene_id']))
#
#     ensembl_ids = []
#     for gene_name in adata.var_names:
#         ensembl_id = gene_symbol_to_ensembl.get(gene_name)
#         if ensembl_id:
#             ensembl_ids.append(ensembl_id)
#         else:
#             ensembl_ids.append(None)
#     adata.var['ensembl_id_grch38'] = ensembl_ids
#
#     # Save rows with None Ensembl IDs to Excel
#     none_mask = adata.var['ensembl_id_grch38'].isnull()
#     none_entries = adata.var[none_mask]
#     none_name_save = os.path.join(save_path, "g38_none_ensembl_ids.xlsx")
#     none_entries.to_excel(none_name_save, index=True)
#     print(f"Entries with None Ensembl IDs for GRCh38 have been saved to: {none_name_save}")
#
#     # Filter out genes that couldn't be mapped
#     adata = adata[:, [eid is not None for eid in ensembl_ids]]
#     return adata
#
#
# # 使用Scanpy加载数据
# adata = sc.read_h5ad(open_path)
#
# # 使用提供的映射文件路径加载基因映射
# mapping_path = os.path.join(args.open_path_conference_data, args.mapping_file)
# gene_symbol_to_ensembl = load_gene_symbol_to_ensembl_mapping(mapping_path)
#
# # 保存无法映射基因的Excel文件路径
# g37_none_name_save = os.path.join(save_path, "g37_none_ensembl_ids.xlsx")
#
# # GRCh37转换
# data_grch37 = convert_gene_names_to_ensembl_grch37(adata.copy(), gene_symbol_to_ensembl, g37_none_name_save)
# output_file_path_grch37 = os.path.join(save_path, "adata_grch37_n_counts.h5ad")
# data_grch37.write(output_file_path_grch37)
# print(f"GRCh37 数据已保存至: {output_file_path_grch37}")
#
# # GRCh38转换
# data_grch38 = convert_gene_names_to_ensembl_grch38(adata.copy())
# output_file_path_grch38 = os.path.join(save_path, "adata_grch38_n_counts.h5ad")
# data_grch38.write(output_file_path_grch38)
# print(f"GRCh38 数据已保存至: {output_file_path_grch38}")