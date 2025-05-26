# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:47:16 2024

@author: qq102
"""



import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import anndata as ad

import argparse
import shap
import random
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
parser.add_argument("--open_path_feature_attribution", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/conference_data', help="Path to open conference data.")


parser.add_argument("--file_prefix", type=str, default='scRNA-seq_panglao_0_1_', help="Prefix for files to handle.")



parser.add_argument('--seed_set', type=int, default=42, help='Number of random seed.')

parser.add_argument('--gene_choose_name', type=str, default='HSP90AA1', help='label of classes.')
parser.add_argument('--feature_name_to_gene', type=str)


parser.add_argument('--jointplot_kwargs_bins', type=int, default=50, help='Number of bins set.')

parser.add_argument('--depleted_ID_len', type=int, default=0, help='Number of the drop the last non-attributable value column.')

parser.add_argument('--Cell_ID', type=str)



parser.add_argument(
    "--shap_indicate_feature",
    type=str,
    nargs='+',  # Allows multiple values
    default=["A1BG", "TP53"]  # Default includes both values
)



# 添加模式参数，类型为字符串，默认值为 'gene_mode'
parser.add_argument(
        '--mode',
        type=str,
        default='gene_mode',
        choices=['gene_mode', 'feature_mode'],
        help='选择运行模式： "gene_mode" 表示基因模式，"feature_mode" 表示特征模式。默认为 "gene_mode"。'
    )

parser.add_argument('--feature_name', type=str,default='scFoundation')


# Parse the arguments
args = parser.parse_args()


# 设置PDF字体参数
plt.rcParams['pdf.fonttype'] = 42  # 确保嵌入字体而不是将其转换为路径

print(args.shap_indicate_feature)


##无torch版
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


set_seed(args.seed_set)



# 检查目录是否存在，如果不存在则创建
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)



##input_data
filenames = os.listdir(args.open_path)
for filename in filenames:
    # scFoundation_Embedding = np.load(os.path.join(open_path, filename))
    full_path = os.path.join(args.open_path, filename)

    # 获取文件后缀名
    _, file_extension = os.path.splitext(filename)

    # 根据不同的文件后缀加载数据
    if file_extension == '.npy':
        scFoundation_Embedding = np.load(full_path)
        print("Loaded .npy file:", scFoundation_Embedding)
    elif file_extension == '.csv':
        scFoundation_Embedding_info = pd.read_csv(full_path)
        print("Loaded .csv file:", scFoundation_Embedding_info)
    elif file_extension == '.xlsx':
        scFoundation_Embedding_info = pd.read_excel(full_path)
        print("Loaded .xlsx file:", scFoundation_Embedding_info)
    elif file_extension == '.h5ad':
        adata = ad.read_h5ad(full_path)
        scFoundation_Embedding = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X  # 确保是NumPy数组
        print("Loaded .h5ad file:", scFoundation_Embedding)
    else:
        print("Unsupported file format")

# 检查数据是否含有NaN
print("DATA is containing NA?: ", np.isnan(scFoundation_Embedding).any())
print('Embedding data shape: ',scFoundation_Embedding.shape)
# print('info data shape: ',scFoundation_Embedding_info.shape)







if args.mode =='gene_mode':

    # 使用绝对路径读入 Excel 文件
    file_path = args.feature_name_to_gene

    data_feature_to_gene = pd.read_excel(file_path)

    # 从第一列生成 feature_names 列表
    feature_names = data_feature_to_gene.iloc[:, 0].tolist()
    # 直接将标签信息添加到 data_df，不设置索引
    data_df = pd.DataFrame(scFoundation_Embedding, columns=feature_names)
elif args.mode =='feature_mode':
    data_df = pd.DataFrame(scFoundation_Embedding)

    # Determine the number of latent dimensions
    latent_dim = data_df.shape[1]

    # Rename the columns to latent_z_num
    data_df.columns = [f'{args.feature_name}_{i}' for i in range(latent_dim)]



##attribution_data_input

# Function to load all CSV and XLSX files in a directory into a dictionary
def load_files_to_dict(directory):
    file_dict = {}
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        file_name, file_extension = os.path.splitext(file)
        if file_extension == '.csv':
            file_dict[file_name] = pd.read_csv(file_path)
        elif file_extension == '.xlsx':
            file_dict[file_name] = pd.read_excel(file_path)
        # file_dict[file_name] = file_dict[file_name].set_index(file_dict[file_name][gene_ID], inplace=True)

    return file_dict



type_data_all=load_files_to_dict(args.open_path_feature_attribution)



for i in type_data_all.keys():
    type_data_all[i]=type_data_all[i].iloc[:,:len(type_data_all[i].columns)-args.depleted_ID_len]




print(f"绘图数据点数量: {len(data_df)}")





class FeaturePlotSource:
    def __init__(self, X_data, shapley_data_all):
        self.X_data = X_data
        self.shapley_data_all = shapley_data_all

    def plot(self):
        pass

class FeaturePlot(FeaturePlotSource):

    class FeatureInteractionAutoChooseDependencePlot:
        def __init__(self, func):
            self.func = func
            self.self_obj = None

        def __get__(self, instance, owner):
            self.self_obj = instance
            return self

        def __call__(self, *args, **kwargs):
            self.plot(*args, **kwargs)

        def plot(self, shap_indicate_feature, save_path, file_name, png_plot=True):
            self.shap_indicate_feature = shap_indicate_feature
            self.save_path = save_path
            self.file_name = file_name
            self.png_plot = png_plot



            for value in list(self.self_obj.shapley_data_all.keys()):
                for i_name in self.shap_indicate_feature:
                    if args.Cell_ID in self.self_obj.shapley_data_all[value].columns:
                        # 原地删除 Cell_ID 列
                        self.self_obj.shapley_data_all[value].drop(columns=[args.Cell_ID], inplace=True)


                    # 将 shapley_data_all 和 X_data 的索引重置为整数
                    self.self_obj.shapley_data_all[value].reset_index(drop=True, inplace=True)
                    self.self_obj.X_data.reset_index(drop=True, inplace=True)
                    # Select only the common columns between the two DataFrames
                    common_columns = self.self_obj.shapley_data_all[value].columns.intersection(self.self_obj.X_data.columns)

                    # Keep only the common columns in both DataFrames
                    self.self_obj.shapley_data_all[value] = self.self_obj.shapley_data_all[value][common_columns]
                    self.self_obj.X_data = self.self_obj.X_data[common_columns]

                    # 创建一个新的图形并设置大小
                    plt.figure(figsize=(10, 10))  # 10 英寸宽，6 英寸高
                    # 绘制 SHAP 特征交互依赖图
                    shap.dependence_plot(i_name, np.array(self.self_obj.shapley_data_all[value]),
                                         self.self_obj.X_data,title=self.file_name+"_"+value+"_"+'shap_dependence' , show=False)
                    if self.png_plot:
                        plt.savefig(f"{args.save_path}/{self.file_name}_{value}_{i_name}_shap_interaction_dependence_plot.png", dpi=300)

                    plt.savefig(f"{args.save_path}/{self.file_name}_{value}_{i_name}_shap_interaction_dependence_plot.pdf", dpi=1000, bbox_inches="tight")


    @FeatureInteractionAutoChooseDependencePlot
    def feature_scatter_plot(self, shap_indicate_feature, save_path,
                             file_name, png_plot, *args, **kwargs):
        pass


test_plot = FeaturePlot(data_df, type_data_all)

test_plot.feature_scatter_plot(
    shap_indicate_feature=args.shap_indicate_feature,
    save_path=args.save_path + '/',
    file_name='',
    png_plot=False
)


# # 假设 args.Cell_ID 是需要删除的列名
# cell_id_column = args.Cell_ID
#
# # 外部循环，处理每个类别的数据
# for value in type_data_all.keys():
#     # 获取每个类别对应的数据
#     shap_values = type_data_all[value]
#     X_data = data_df
#
#     # 删除 Cell_ID 列
#     if cell_id_column in shap_values.columns:
#         shap_values.drop(columns=[cell_id_column], inplace=True)
#
#     if cell_id_column in X_data.columns:
#         X_data.drop(columns=[cell_id_column], inplace=True)
#
#     # 重置索引为整数索引
#     shap_values.reset_index(drop=True, inplace=True)
#     X_data.reset_index(drop=True, inplace=True)
#
#     # 检查列是否删除成功并打印确认
#     print(f"Processing value: {value}")
#     print(f"SHAP values shape after dropping {cell_id_column}: {shap_values.shape}")
#     print(f"X_data shape after dropping {cell_id_column}: {X_data.shape}")
#
#     # 确保索引一致后绘制 SHAP 特征交互依赖图
#     for i_name in shap_values.columns:
#         shap.dependence_plot(i_name, shap_values, X_data, show=False)
#
#         # 保存 SHAP 图像
#         plt.savefig(f"{args.save_path}/{args.file_name}_{value}_{i_name}_shap_interaction_dependence_plot.png", dpi=300)
#         plt.savefig(f"{args.save_path}/{args.file_name}_{value}_{i_name}_shap_interaction_dependence_plot.pdf",
#                     dpi=1000, bbox_inches="tight")
#         plt.clf()  # 清空当前图形，以便下一个图形的绘制