
## Global parameter tuning
import os
import pandas as pd
import numpy as np

from protloc_mex1.SHAP_plus import SHAP_importance_sum
import re
import anndata as ad
from sklearn.preprocessing import MinMaxScaler
import pickle
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
import math
from scipy.stats import mannwhitneyu, kruskal
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
parser.add_argument("--feature_attribution_value_path", type=str)
parser.add_argument("--save_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/output/VAE_inference', help="Path to save output data.")
parser.add_argument("--gene_shift", type=strict_str2bool, default=False, help="if need ENSGxxxx convert to gene name")
parser.add_argument("--file_prefix", type=str, default='scRNA-seq_panglao_0_1_', help="Prefix for files to handle.")
parser.add_argument('--gene_ENG_mapping_file', type=str,default= 'mart_export.txt',help= 'file path of ENSGxxxx convert to gene name')
parser.add_argument('--gene_ID', type=str, default='Ensembl_ID', help='label of classes.')
parser.add_argument('--feature_name_to_gene', type=str,help= 'file path of scfoundation_19264_gene_index.xlsx')

parser.add_argument("--auto_indicate", type=strict_str2bool, default=False)

parser.add_argument("--key_name", type=str, default='feature')

parser.add_argument("--drug_name", type=str, default='Gefitinib')

parser.add_argument("--positive_color", type=str, default='#FFB6C1')
parser.add_argument("--negative_color", type=str, default='#ADD8E6')
parser.add_argument("--plot_gene_num", type=int, default=10)

# 用户可以用逗号分隔输入多个基因名，例如：
#   python script.py --select_genes "GeneA,GeneB,GeneC"
parser.add_argument('--select_genes', type=str, default='',
                    help='Comma-separated list of genes to plot, e.g. "GeneA,GeneB,GeneC"')

# parser.add_argument("--heatmapcolor", type=str, default='YlGn')
parser.add_argument("--all_heatmap_plot", type=strict_str2bool, default=False)
parser.add_argument("--probability_stage_boxplot", type=strict_str2bool, default=False)

# Parse the arguments
args = parser.parse_args()



class FeaturePlotSource:
    def __init__(self, X_data, shapley_data_all):
        self.X_data = X_data
        self.shapley_data_all = shapley_data_all

    def plot(self):
        pass


class LocalAnalysisPlot(FeaturePlotSource):
    def __init__(self, X_data, shapley_data_all, probability_data):
        super().__init__(X_data, shapley_data_all)
        self.probability_data = probability_data

    def plot(self, protein_indicate, save_path, file_name='huamn', plot_size=(10, 10), png_plot=True,
             feature_remain=5, positive_color='#FFB6C1', negative_color='#ADD8E6'):
        # protein_indicate= protein_indicate
        # save_path = save_path
        # file_name = file_name
        # png_plot = png_plot

        for value in list(self.shapley_data_all.keys()):
            for i_name in protein_indicate:
                ##创建绘图概率
                probability_predict = self.probability_data[value].loc[i_name].to_numpy()[0]

                ##创建绘图列表
                df = pd.DataFrame({'Feature_value': self.X_data.loc[i_name],
                                   'shapley_value': self.shapley_data_all[value].loc[i_name]})
                df['shapley_value_abs'] = df['shapley_value'].abs()
                df = df.sort_values(by='shapley_value_abs', ascending=False).iloc[:feature_remain]
                df['abs_rank'] = df['shapley_value_abs'].rank(ascending=True)
                feature_name = df.index.tolist()
                feature_value = df['Feature_value'].tolist()
                yticklabels = [f"{feature_name[i]} ({feature_value[i]:.4f})" for i in range(len(feature_name))]
                # 将正数和负数数据分别存储到两个不同的数组中
                # 使用条件语句将正数和负数数据分别存储到两个不同的DataFrame中
                positive_data = df[df['shapley_value'] > 0]
                negative_data = df[df['shapley_value'] <= 0]
                plt.rcParams['pdf.fonttype'] = 42

                # 创建一个包含两个水平直方图的子图
                fig, ax = plt.subplots()

                # 绘制正数数据的水平直方图
                ax.barh(positive_data['abs_rank'].to_numpy(),
                        positive_data['shapley_value'], align='center', color=positive_color)

                # 绘制负数数据的水平直方图
                ax.barh(negative_data['abs_rank'].to_numpy(),
                        negative_data['shapley_value'], align='center', color=negative_color)
                ax.set_yticks(df['abs_rank'].to_numpy())
                ax.set_yticklabels(yticklabels)

                # 添加概率预测值的注释
                ax.text(1, 1, f"Based on {i_name} gene expression value, predicting {args.drug_name} {value} probability is: {probability_predict:.2f}",
                        transform=ax.transAxes,
                        verticalalignment='bottom', horizontalalignment='right')
                ax.set_xlabel('IG_value')  ##x轴标题
                ax.set_ylabel(f"Gene expression value of {i_name}") ##y轴标题

                fig.set_size_inches(plot_size)
                fig.savefig(save_path +
                            f"{file_name}_{value}_{i_name}_localplot.pdf",
                            dpi=1000, bbox_inches="tight")
                if png_plot:
                    fig.savefig(save_path +
                                f"{file_name}_{value}_{i_name}_localplot.png",
                                dpi=1000, bbox_inches="tight")
                plt.close(ax.figure)







def process_data(keyword,names,open_path):
    # Find file names containing a specific keyword and remove the '.csv' extension
    file_names = SHAP_importance_sum.str_contain(f'{keyword}_shap_values', names)
    file_names = [name.replace('.csv', '') for name in file_names]

    # Assume there's only one element in file_names
    file_name = file_names[0]

    # Create the full file path
    file_path = os.path.join(open_path, f'{file_name}.csv')

    # Read the CSV file using pandas
    df = pd.read_csv(file_path,index_col=0,header=0)

    return df




# This section processes the results of Random Forest (RF). If you need to process results for DNN or DNN_ig,
# simply change the 'open_path' variable to the corresponding directory.
open_path = args.open_path
feature_attribution_value_path = args.feature_attribution_value_path
save_path = args.save_path



# 检查路径是否存在
if not os.path.exists(save_path):
    # 如果路径不存在，则创建该路径
    os.makedirs(save_path)

names = os.listdir(feature_attribution_value_path)

# Read the target file based on the file keyword
df_feature_attribution = process_data(args.key_name,names,feature_attribution_value_path)

df_feature_attribution_score = df_feature_attribution.copy()

# 1. 计算每行的和
df_feature_attribution_score['row_sum'] = df_feature_attribution_score.sum(axis=1)

# 2. 对求和结果应用 Min-Max 归一化
scaler = MinMaxScaler()
df_feature_attribution_score['row_sum_scaled'] = scaler.fit_transform(df_feature_attribution_score[['row_sum']])

##将min-max后的求和特征归因值解释为概率，不推荐，因为没有考虑基线值
Probability_data = df_feature_attribution_score['row_sum_scaled']

##将min-max后的求和特征归因值解释为概率，不推荐，因为没有考虑基线值
Probability_data = df_feature_attribution_score['row_sum_scaled']


# 选择需要保存的列
df_feature_attribution_score[['row_sum', 'row_sum_scaled']].to_csv(
    os.path.join(save_path, f'{args.key_name}_scoring.csv'),
    index=True
)

print(f"处理后的 DataFrame 已保存到: {save_path}")

# 保存 Min-Max 归一化参数
min_max_params = {
    'min': scaler.data_min_[0],
    'max': scaler.data_max_[0],
    'scale': scaler.scale_[0],
    'feature_range': scaler.feature_range
}

params_path = os.path.join(save_path, 'minmax_params.pkl')

with open(params_path, 'wb') as f:
    pickle.dump(min_max_params, f)


print(f"Min-Max 归一化参数已保存到: {params_path}")




#######################################################
#             读入生存数据和病人的表达量                   #
#######################################################



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

import pandas as pd
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


# 文件路径

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


survival_data.to_excel(os.path.join(save_path, 'survival_data.xlsx'))

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


# 检查维度是否相同
if X_df.shape == df_feature_attribution.shape:
    print("两个 DataFrame 的维度相同。")
else:
    print("维度不同：")
print(f"X_df 的维度为 {X_df.shape}")
print(f"df_feature_attribution 的维度为 {df_feature_attribution.shape}")

# 检查列名是否相同且顺序一致
if list(X_df.columns) == list(df_feature_attribution.columns):
    print("两个 DataFrame 的列名顺序相同。")
else:
    print("列名顺序不同。")

# 检查索引是否相同
if X_df.index.equals(df_feature_attribution.index):
    print("两个 DataFrame 的索引相同。")
else:
    print("索引不同。")
print("X_df 的索引：", X_df.index)
print("df_feature_attribution 的索引：", df_feature_attribution.index)

# print("X_df 的列名：")
# print(X_df.columns.tolist())

# # 打印 df_feature_attribution 的列名
# print("\ndf_feature_attribution 的列名：")
# print(df_feature_attribution.columns.tolist())

df_feature_attribution.index = X_df.index





#
# # Create the DataFrame of feature value
# X_input = pd.DataFrame(scFoundation_Embedding, index=ids, columns=feature_names)








dict_feature_attribution = {args.key_name:df_feature_attribution}

# 如果是 Series，转换为 DataFrame
if isinstance(Probability_data, pd.Series):
    Probability_data = Probability_data.to_frame()
    print("已将 Probability_data 从 Series 转换为 DataFrame。")
elif isinstance(Probability_data, pd.DataFrame):
    print("Probability_data 已经是 DataFrame。")
else:
    raise TypeError("Probability_data 必须是 pandas.Series 或 pandas.DataFrame。")


# 检查行数是否匹配
if X_df.shape[0] != Probability_data.shape[0]:
    raise ValueError(f"行数不匹配：X_df 有 {X_df.shape[0]} 行，Probability_data 有 {Probability_data.shape[0]} 行。")
else:
    print("X_df 和 Probability_data 的行数匹配。")

# 检查并设置索引
Probability_data.index = X_df.index
print("已将 Probability_data 的索引设置为 X_df 的索引。")

# 验证索引对齐
if Probability_data.index.equals(X_df.index):
    print("Probability_data 的索引已成功对齐为 X_df 的索引。")
else:
    print("Probability_data 的索引未能对齐为 X_df 的索引。请检查数据。")



Probability_data_dict = {args.key_name:Probability_data}




if args.auto_indicate:

    def validate_columns(df, required_columns):
        """验证 DataFrame 是否包含所需的列。"""
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            print(f"数据中缺少列: {', '.join(missing)}。请检查数据格式。")
            return False
        print(f"数据包含必要的列: {', '.join(required_columns)}。")
        return True


    def set_sample_as_index(df, sample_column='sample'):
        """将指定的列设置为 DataFrame 的索引。"""
        if df[sample_column].duplicated().any():
            print(f"警告：'{sample_column}' 列中存在重复值，可能导致索引设置问题。")
        df = df.set_index(sample_column)
        print(f"'{sample_column}' 列已成功设置为索引。")
        return df


    def filter_samples(df, reference_index):
        """仅保留索引在 reference_index 中存在的行。"""
        initial_count = len(df)
        df_filtered = df[df.index.isin(reference_index)]
        filtered_count = len(df_filtered)
        print(f"已过滤生存数据：从 {initial_count} 行减少到 {filtered_count} 行，仅保留在 X_df.index 中存在的样本。")
        return df_filtered


    def split_survival_data(df, os_column='OS', time_column='OS.time', top_n=3):
        """根据 'OS' 和 'OS.time' 列，分割 DataFrame 为四个子集。"""
        os_0 = df[df[os_column] == 0]
        os_1 = df[df[os_column] == 1]
        # print(f"OS = 0 的样本数: {len(os_0)}")
        # print(f"OS = 1 的样本数: {len(os_1)}")

        if len(os_0) < top_n or len(os_1) < top_n:
            print("某个子集的样本数少于3，无法进行分割。")
            return None, None, None, None

        os_0_max3 = os_0.sort_values(by=time_column, ascending=False).head(top_n).copy()
        os_0_min3 = os_0.sort_values(by=time_column, ascending=True).head(top_n).copy()
        os_1_max3 = os_1.sort_values(by=time_column, ascending=False).head(top_n).copy()
        os_1_min3 = os_1.sort_values(by=time_column, ascending=True).head(top_n).copy()

        return os_0_max3, os_0_min3, os_1_max3, os_1_min3


    def save_to_excel(file_path, dataframes, sheet_names):
        """将多个 DataFrame 保存到同一个 Excel 文件的不同工作表中。"""
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            for df, sheet in zip(dataframes, sheet_names):
                if df is not None:
                    df.to_excel(writer, sheet_name=sheet, index=False)
        print(f"分割后的生存数据已成功保存为 Excel 文件：{file_path}")


    def create_groups_dict(dataframes, group_names):
        """创建一个字典，将组名映射到相应的 DataFrame。"""
        groups = {name: df for name, df in zip(group_names, dataframes) if df is not None}
        return groups


    survival_data_scoring = survival_data.copy()
    # 1. 验证必要的列
    if not validate_columns(survival_data_scoring, ['sample', 'OS', 'OS.time']):
        exit()

    # 2. 设置 'sample' 列为索引
    survival_data_scoring = set_sample_as_index(survival_data_scoring, 'sample')

    # 3. 仅保留在 X_df.index 中存在的样本
    survival_data_scoring = filter_samples(survival_data_scoring, X_df.index)

    # 4. 分割生存数据
    os_0_max3, os_0_min3, os_1_max3, os_1_min3 = split_survival_data(
        survival_data_scoring, os_column='OS', time_column='OS.time', top_n=3
    )

    # 7. 定义保存的 Excel 文件路径
    excel_file_path = os.path.join(save_path, 'auto_choose_survival_data.xlsx')

    # 8. 保存到 Excel
    save_to_excel(
        excel_file_path,
        [os_0_max3, os_0_min3, os_1_max3, os_1_min3],
        ['OS0_Max3', 'OS0_Min3', 'OS1_Max3', 'OS1_Min3']
    )

    # 9. 创建一个字典，映射组名到相应的 DataFrame
    groups = {
        'OS0_Max3': os_0_max3,
        'OS0_Min3': os_0_min3,
        'OS1_Max3': os_1_max3,
        'OS1_Min3': os_1_min3
    }



    indicate_feature_plot = LocalAnalysisPlot(X_df,dict_feature_attribution,Probability_data_dict)

    # 遍历每个组，提取 sample ID 并调用 plot 函数
    for group_name, df in groups.items():
        # 提取 sample ID 列（假设列名为 'sample'）
        sample_ids = df.index.tolist()

        # 根据组名生成文件名，例如 'OS0_Max3_plot.png'
        file_name = f'{group_name}'

        # 调用 plot 方法
        indicate_feature_plot.plot(
            sample_ids,  # 传递 sample ID 列表
            save_path+'/',  # 保存路径
            file_name=file_name,  # 文件名
            plot_size=(10, 10),  # 绘图尺寸
            png_plot=True,  # 是否保存为 PNG 格式
            feature_remain=args.plot_gene_num,  # 保留特征数量
            positive_color=args.positive_color,  # 正向颜色
            negative_color=args.negative_color # 负向颜色
        )

        print(f"已保存 {group_name} 的图形为 {file_name}")



# if args.patient_indicate:
#     '''
#     developing
#     '''


if args.all_heatmap_plot:

    def validate_columns(df, required_columns):
        """验证 DataFrame 是否包含所需的列。"""
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            print(f"数据中缺少列: {', '.join(missing)}。请检查数据格式。")
            return False
        print(f"数据包含必要的列: {', '.join(required_columns)}。")
        return True


    def set_sample_as_index(df, sample_column='sample'):
        """将指定的列设置为 DataFrame 的索引。"""
        if df[sample_column].duplicated().any():
            print(f"警告：'{sample_column}' 列中存在重复值，可能导致索引设置问题。")
        df = df.set_index(sample_column)
        print(f"'{sample_column}' 列已成功设置为索引。")
        return df


    def filter_samples(df, reference_index):
        """仅保留索引在 reference_index 中存在的行。"""
        initial_count = len(df)
        df_filtered = df[df.index.isin(reference_index)]
        filtered_count = len(df_filtered)
        print(f"已过滤生存数据：从 {initial_count} 行减少到 {filtered_count} 行，仅保留在 X_df.index 中存在的样本。")
        return df_filtered


    survival_data_heatmap = survival_data.copy()

    # 验证必要的列
    if not validate_columns(survival_data_heatmap, ['sample', 'OS', 'OS.time']):
        exit()


    # 设置 'sample' 列为索引
    survival_data_heatmap = set_sample_as_index(survival_data_heatmap, 'sample')

    # 仅保留在 X_df.index 中存在的样本
    survival_data_heatmap = filter_samples(survival_data_heatmap, X_df.index)

    # 4. 分割生存数据
    OS0 = survival_data_heatmap[survival_data_heatmap['OS'] == 0].sort_values(by='OS.time')
    OS1 = survival_data_heatmap[survival_data_heatmap['OS'] == 1].sort_values(by='OS.time')

    # 5. 绘制热图

    # 处理基因列表
    # 如果用户提供了 select_genes，那么将其 split 成列表；如果为空就默认为 None 或空列表
    if args.select_genes:
        selected_genes = args.select_genes.split(',')
        # 去除每个基因前后空白
        selected_genes = [g.strip() for g in selected_genes]
    else:
        selected_genes = []

    # 根据 select_genes 进行列筛选
    if selected_genes:
        # 保证这些基因在 X_df 中
        selected_genes_in_data = [gene for gene in selected_genes if gene in X_df.columns]
        X_df = X_df.loc[:, selected_genes_in_data]

        # 同样对归因矩阵也进行列筛选
        selected_genes_in_attr = [gene for gene in selected_genes if gene in df_feature_attribution.columns]
        df_feature_attribution = df_feature_attribution.loc[:, selected_genes_in_attr]

    cmap = LinearSegmentedColormap.from_list(
        "BlueWhiteRed",
        [args.negative_color, "white", args.positive_color]
    )

    # # 定义颜色映射
    # cmap = args.heatmapcolor  # 黄绿渐变色

    # 创建绘图对象
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # 获取数据
    gene_expression = X_df.copy()

   
    # 绘制OS=0的热图
    sns.heatmap(gene_expression.loc[OS0.index], cmap=cmap, ax=axes[0],center=0)
    axes[0].set_title('OS=0')

    # 绘制OS=1的热图
    sns.heatmap(gene_expression.loc[OS1.index], cmap=cmap, ax=axes[1],center=0)
    axes[1].set_title('OS=1')

    # 调整子图间距
    plt.tight_layout()

    # 保存图形
    plt.savefig(os.path.join(save_path, 'gene_expression_heatmap_plots.pdf'),dpi=1000, bbox_inches="tight")
    plt.close(fig)

   ### 特征归因热图

    # 3. 创建绘图对象
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # 4. 分别绘制 OS=0 和 OS=1 的特征归因热图
    sns.heatmap(
        df_feature_attribution.loc[OS0.index],  # 使用 OS=0 组对应的行索引
        cmap=cmap,
        ax=axes[0],
        center=0
    )
    axes[0].set_title('Feature Attribution (OS=0)')

    sns.heatmap(
        df_feature_attribution.loc[OS1.index],  # 使用 OS=1 组对应的行索引
        cmap=cmap,
        ax=axes[1],
        center=0
    )
    axes[1].set_title('Feature Attribution (OS=1)')

    # 5. 调整子图间距
    plt.tight_layout()

    # 6. 保存图形
    plt.savefig(os.path.join(save_path, 'feature_attribution_heatmap_by_OS.pdf'),
                dpi=1000, bbox_inches="tight")
    plt.close(fig)


    # ============== 2. 绘制小提琴图并在其中做统计检验 ==============
    pdf_path = os.path.join(save_path, 'all_genes_one_page_violin.pdf')
    genes_to_plot = selected_genes  # 或者 selected_genes_in_data, 视情况而定

    # 如果你想要每行放多少个子图，就设定 ncols
    ncols = 5
    # 计算行数
    nrows = math.ceil(len(genes_to_plot) / ncols)

    # 创建图形和子图
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()  # 将二维数组变为一维，方便逐个访问

    # 用于存储检验结果
    stats_results = []

    for i, gene in enumerate(genes_to_plot):
        ax = axes[i]

        # 去除重复列（如有）
        X_df = X_df.loc[:, ~X_df.columns.duplicated()]

        # 构建临时 DF
        temp_df = pd.concat([X_df[[gene]], survival_data_heatmap[['OS']]], axis=1)

        # 打印调试信息(可选)
        # print(temp_df)
        # print(temp_df.columns)

        # 重新命名列，方便后面绘图和检验
        temp_df.columns = ['expression', 'OS']

        # ② 分组取值
        group0 = temp_df.loc[temp_df['OS'] == 0, 'expression']
        group1 = temp_df.loc[temp_df['OS'] == 1, 'expression']

        # ③ Mann-Whitney U (Wilcoxon) 检验
        stat, pval = mannwhitneyu(group0, group1, alternative='two-sided')

        # ④ 小提琴图
        sns.violinplot(
            x='OS',
            y='expression',
            data=temp_df,
            ax=ax,
            inner='quartile',
            palette=[args.negative_color, args.positive_color]  # <-- 在这里指定你想要的两个颜色
        )

        # 在图标题中显示基因名和 p 值(科学计数法，保留3位指数)
        ax.set_title(f"{gene}\np={pval:.3e}")

        # 存储结果（可选）
        stats_results.append({
            'gene': gene,
            'stat': stat,
            'pval': pval,
            'median_OS0': np.median(group0),
            'median_OS1': np.median(group1),
            'n_OS0': len(group0),
            'n_OS1': len(group1)
        })

    # 如果最后一行/列子图不够用，需要隐藏空白子图
    for j in range(len(genes_to_plot), nrows * ncols):
        axes[j].axis('off')

    plt.tight_layout()

    # 将此页保存到 PDF
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close(fig)

    print(f"所有基因的小提琴图已保存在: {pdf_path}")

    # ============== 3. 保存统计检验结果到 CSV (可选) ==============
    results_df = pd.DataFrame(stats_results)
    results_csv_path = os.path.join(save_path, 'wilcox_test_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"Wilcoxon 检验结果已保存到: {results_csv_path}")




if args.probability_stage_boxplot:

    def validate_columns(df, required_columns):
        """验证 DataFrame 是否包含所需的列。"""
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            print(f"数据中缺少列: {', '.join(missing)}。请检查数据格式。")
            return False
        print(f"数据包含必要的列: {', '.join(required_columns)}。")
        return True


    def set_sample_as_index(df, sample_column='sample'):
        """将指定的列设置为 DataFrame 的索引。"""
        if df[sample_column].duplicated().any():
            print(f"警告：'{sample_column}' 列中存在重复值，可能导致索引设置问题。")
        df = df.set_index(sample_column)
        print(f"'{sample_column}' 列已成功设置为索引。")
        return df


    def filter_samples(df, reference_index):
        """仅保留索引在 reference_index 中存在的行。"""
        initial_count = len(df)
        df_filtered = df[df.index.isin(reference_index)]
        filtered_count = len(df_filtered)
        print(f"已过滤生存数据：从 {initial_count} 行减少到 {filtered_count} 行，仅保留在 X_df.index 中存在的样本。")
        return df_filtered


    survival_data_heatmap = survival_data.copy()

    # 验证必要的列
    if not validate_columns(survival_data_heatmap, ['sample', 'OS', 'OS.time']):
        exit()


    # 设置 'sample' 列为索引
    survival_data_heatmap = set_sample_as_index(survival_data_heatmap, 'sample')

    # 仅保留在 X_df.index 中存在的样本
    survival_data_heatmap = filter_samples(survival_data_heatmap, X_df.index)

    # 1. 从 survival_data_heatmap 中筛选出 OS=1 的样本
    df_OS1 = survival_data_heatmap[survival_data_heatmap['OS'] == 1].copy()

    # # 2. 使用 pd.qcut 分箱，并返回分箱边界
    # df_OS1['time_bin'], bin_edges = pd.qcut(
    #     df_OS1['OS.time'],
    #     q=5,
    #     labels=[f"Bin{i}" for i in range(1, 6)],
    #     retbins=True,  # 开启后会额外返回分箱的边界
    #     duplicates='drop'  # 如果出现重复边界可去重
    # )

    # 2. 使用 pd.cut 进行等宽分箱，并返回分箱边界
    df_OS1['time_bin'], bin_edges = pd.cut(
        df_OS1['OS.time'],
        bins=10,  # 等宽分为10段
        labels=[f"Bin{i}" for i in range(1, 11)],
        retbins=True  # 返回分箱边界
    )
    # bin_edges 长度为 11，分别是这10个区间的边界
    print("等宽分箱边界：", bin_edges)

    print("分位箱边界：", bin_edges)
    # bin_edges 是一个数组，其中包含 6 个数值，分别对应 [time_min, Q1, Q2, Q3, Q4, time_max]

    #  绘制直方图
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df_OS1, x='OS.time', bins=30, kde=False, color=args.negative_color)

    # 在直方图上用竖线标记 5 分位点
    #   bin_edges[0] 是最小值，bin_edges[-1] 是最大值，一般我们标记中间的几个点即可
    for edge in bin_edges[1:-1]:
        plt.axvline(edge, color=args.positive_color, linestyle='--', alpha=0.7)
        edge_int = round(edge)
        # 若想标出数值，可额外加上文本:
        plt.text(edge, plt.ylim()[1] * 0.9, f"{edge_int:d}", color=args.positive_color,
                 ha='center', va='top', rotation=90)

    plt.title("OS-1")
    plt.xlabel("OS.time")
    plt.ylabel("Count")

    plt.tight_layout()

    # 保存图形
    plt.savefig(os.path.join(save_path, "os1_time_hist_with_quantiles.pdf"), dpi=1000)
    plt.close()


    # 3. 将特征归因值求和归一化分数 (row_sum_scaled) 合并到 df_OS1
    #   假设 df_feature_attribution_score 的行索引也是样本 ID

    df_feature_attribution_score_for_bin_plot = df_feature_attribution_score.copy()
    # 检查行数是否匹配
    if X_df.shape[0] != df_feature_attribution_score_for_bin_plot.shape[0]:
        raise ValueError(f"行数不匹配：X_df 有 {X_df.shape[0]} 行，df_feature_attribution_score_for_bin_plot 有 {df_feature_attribution_score_for_bin_plot.shape[0]} 行。")
    else:
        print("X_df 和 df_feature_attribution_score_for_bin_plot 的行数匹配。")

    # 检查并设置索引
    df_feature_attribution_score_for_bin_plot.index = X_df.index
    print("已将 df_feature_attribution_score_for_bin_plot 的索引设置为 X_df 的索引。")

    # 验证索引对齐
    if df_feature_attribution_score_for_bin_plot.index.equals(X_df.index):
        print("df_feature_attribution_score_for_bin_plot 的索引已成功对齐为 X_df 的索引。")
    else:
        print("df_feature_attribution_score_for_bin_plot 的索引未能对齐为 X_df 的索引。请检查数据。")

    df_OS1 = df_OS1.join(df_feature_attribution_score_for_bin_plot[['row_sum_scaled']], how='inner')

    # 检查合并后是否包含所需列
    print("df_OS1 columns:", df_OS1.columns)

    # 4. 绘制箱线图
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df_OS1,
        x='time_bin',
        y='row_sum_scaled',
        boxprops=dict(facecolor='none', edgecolor=args.negative_color, linewidth=1),  # 设置箱子的属性
        medianprops=dict(color=args.negative_color, linewidth=1),  # 设置中位线的属性
        whiskerprops=dict(color=args.negative_color, linewidth=1),  # 设置须的属性
        capprops=dict(color=args.negative_color, linewidth=1),  # 设置帽的属性
        flierprops=dict(markeredgecolor=args.negative_color, marker='o', markersize=1),  # 设置离群点的属性
        linewidth=1.5  # 设置整体线宽
    )


    plt.title("Boxplot of row_sum_scaled by OS.time bins (OS=1 only)")
    plt.ylabel(f"{args.key_name} probability")
    plt.xlabel("OS.time bins")

    # 5. 保存图形
    plt.tight_layout()
    plot_save_path = os.path.join(save_path, "OS1_time_bin_boxplot.pdf")
    plt.savefig(plot_save_path, dpi=1000)
    plt.close()

    print(f"OS=1 的 5 组分箱箱线图已保存到: {plot_save_path}")




#
# #####################################
# # 1. 数据准备与验证
# #####################################
#
# def validate_columns(df, required_columns):
#     """验证 DataFrame 是否包含所需的列。"""
#     missing = [col for col in required_columns if col not in df.columns]
#     if missing:
#         print(f"数据中缺少列: {', '.join(missing)}。请检查数据格式。")
#         return False
#     print(f"数据包含必要的列: {', '.join(required_columns)}。")
#     return True
#
# def set_sample_as_index(df, sample_column='sample'):
#     """
#     将指定的列设置为 DataFrame 的索引。
#     如果存在重复值，会打印警告信息。
#     """
#     if df[sample_column].duplicated().any():
#         print(f"警告：'{sample_column}' 列中存在重复值，可能导致索引设置问题。")
#     df = df.set_index(sample_column)
#     print(f"'{sample_column}' 列已成功设置为索引。")
#     return df
#
# def filter_samples(df, reference_index):
#     """
#     仅保留索引在 reference_index 中存在的行。
#     通常用来将生存数据和表达矩阵对齐。
#     """
#     initial_count = len(df)
#     df_filtered = df[df.index.isin(reference_index)]
#     filtered_count = len(df_filtered)
#     print(f"已过滤生存数据：从 {initial_count} 行减少到 {filtered_count} 行，仅保留在 X_df.index 中存在的样本。")
#     return df_filtered
#
#
# #####################################
# # 主功能逻辑：分三组 & 绘图
# #####################################
#
# # 处理基因列表
# if args.select_genes:
#     selected_genes = args.select_genes.split(',')
#     # 去除每个基因前后空白
#     selected_genes = [g.strip() for g in selected_genes]
# else:
#     selected_genes = []
#
# # 1) 拷贝一份 survival_data 以备热图使用
# survival_data_heatmap = survival_data.copy()
#
# # 2) 验证必要的列
# if not validate_columns(survival_data_heatmap, ['sample', 'OS', 'OS.time']):
#     raise SystemExit("生存数据不完整，退出脚本。")
#
# # 3) 设置 'sample' 列为索引
# survival_data_heatmap = set_sample_as_index(survival_data_heatmap, 'sample')
#
# # 4) 过滤到与 X_df 同索引的样本
# survival_data_heatmap = filter_samples(survival_data_heatmap, X_df.index)
#
# #####################################
# # 基于 OS=1 的中位生存时间, 划分三组
# #####################################
#
# # OS=1 -> 分两组: OS1_low, OS1_high
# # OS=0 -> 仅保留 OS.time >= median_OS1 -> OS0_keep；其余样本剔除
#
# OS1 = survival_data_heatmap[survival_data_heatmap['OS'] == 1]
# OS0 = survival_data_heatmap[survival_data_heatmap['OS'] == 0]
#
# median_OS1 = OS1['OS.time'].median()
# print(f"OS=1 组中位生存时间: {median_OS1:.2f}")
#
# # 新增列 group_label
# survival_data_heatmap['group_label'] = None
#
# # OS1_low
# mask_OS1_low = (survival_data_heatmap['OS'] == 1) & (survival_data_heatmap['OS.time'] < median_OS1)
# survival_data_heatmap.loc[mask_OS1_low, 'group_label'] = 'OS1_low'
#
# # OS1_high
# mask_OS1_high = (survival_data_heatmap['OS'] == 1) & (survival_data_heatmap['OS.time'] >= median_OS1)
# survival_data_heatmap.loc[mask_OS1_high, 'group_label'] = 'OS1_high'
#
# # OS0_keep
# mask_OS0_keep = (survival_data_heatmap['OS'] == 0) & (survival_data_heatmap['OS.time'] >= median_OS1)
# survival_data_heatmap.loc[mask_OS0_keep, 'group_label'] = 'OS0_keep'
#
# initial_count = len(survival_data_heatmap)
# # 剔除不属于以上三组的样本 (尤其是 OS=0 & OS.time < median_OS1)
# survival_data_heatmap = survival_data_heatmap.dropna(subset=['group_label'])
# final_count = len(survival_data_heatmap)
# print(f"剔除了 OS=0 中 OS.time < {median_OS1:.2f} 的样本 {initial_count - final_count} 个。")
#
# print("三组分布：")
# print(survival_data_heatmap['group_label'].value_counts())
#
# #####################################
# # 对 X_df / df_feature_attribution 根据 selected_genes 做筛选
# #####################################
# if selected_genes:
#     # 只保留在 X_df.columns 中的基因
#     selected_genes_in_data = [gene for gene in selected_genes if gene in X_df.columns]
#     X_df = X_df.loc[:, selected_genes_in_data]
#
#     # 对归因矩阵同样处理
#     selected_genes_in_attr = [gene for gene in selected_genes if gene in df_feature_attribution.columns]
#     df_feature_attribution = df_feature_attribution.loc[:, selected_genes_in_attr]
# else:
#     # 若没提供基因，则可以根据需求决定不画图或给出提示
#     print("未指定 select_genes，将不执行基因相关热图和小提琴图。")
#
#
# #####################################
# # 3. 绘制热图 (只用选定的基因)
# #####################################
# cmap = LinearSegmentedColormap.from_list("BlueWhiteRed", ["blue", "white", "red"])
#
# groups = ['OS1_low', 'OS1_high', 'OS0_keep']
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(24, 6))
#
# for i, g in enumerate(groups):
#     idx = survival_data_heatmap[survival_data_heatmap['group_label'] == g].index
#     # 只取这一组 + 只保留选定基因后的表达矩阵
#     data_sub = X_df.loc[idx].copy()
#     sns.heatmap(data_sub, cmap=cmap, ax=axes[i], center=0)
#     axes[i].set_title(f"{g} (n={len(data_sub)})")
#
# plt.tight_layout()
# heatmap_path = os.path.join(save_path, "three_group_subplots_heatmap.pdf")
# plt.savefig(heatmap_path, dpi=300)
# plt.close(fig)
# print(f"三组热图已保存到: {heatmap_path}")
#
# #####################################
# # 4. 绘制三组小提琴图 (只用选定的基因)
# #####################################
# # 构建合并表：只保留三组样本
# X_df_filtered = X_df.loc[survival_data_heatmap.index].copy()
# # 在合并好的 DataFrame 中加入 'group_label'
# X_df_annot = X_df_filtered.join(survival_data_heatmap[['group_label']], how='left')
#
# # 如果此时 X_df_annot 为空或选基因太少，检查一下
# if X_df_annot.empty or len(X_df_annot.columns) <= 1:
#     print("X_df_annot 为空或没有可绘制的基因，无法绘制小提琴图。")
#
#
# violin_pdf_path = os.path.join(save_path, "three_group_violin.pdf")
# ncols = 4  # 每行放4个小提琴
# nrows = math.ceil(len(selected_genes) / ncols)
#
# fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
# axes = axes.flatten()
#
# # 保存统计检验结果
# stats_results = []
#
# for i, gene in enumerate(selected_genes):
#     ax = axes[i]
#     # 构造临时数据：两列 [expression, group_label]
#     temp_df = X_df_annot[[gene, 'group_label']].dropna()
#     #去除重复列（如有）
#     temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()]
#     temp_df.columns = ['expression', 'group_label']
#
#     # 如果这一基因在三组中都没有数据，跳过
#     if temp_df.empty:
#         ax.set_title(f"{gene}\n无可用数据")
#         ax.axis('off')
#         continue
#
#     # 画小提琴图 (三组: OS1_low, OS1_high, OS0_keep)
#     sns.violinplot(
#         x='group_label',
#         y='expression',
#         data=temp_df,
#         ax=ax,
#         order=['OS1_low','OS1_high','OS0_keep'],  # 三组顺序
#         inner='quartile'
#     )
#     ax.set_title(gene)
#
#     # 统计检验: Kruskal-Wallis（多组比较）
#     group_low = temp_df.loc[temp_df['group_label'] == 'OS1_low', 'expression']
#     group_high = temp_df.loc[temp_df['group_label'] == 'OS1_high', 'expression']
#     group_os0 = temp_df.loc[temp_df['group_label'] == 'OS0_keep', 'expression']
#
#     # 若有些组为空，会导致 kruskal 出错，这里做个简单保护
#     non_empty_groups = [gr for gr in [group_low, group_high, group_os0] if len(gr) > 0]
#     if len(non_empty_groups) < 2:
#         ax.set_xlabel("只有1组有数据，无法统计检验")
#         pval = np.nan
#         stat = np.nan
#     else:
#         stat, pval = kruskal(group_low, group_high, group_os0)
#         ax.set_xlabel(f"Kruskal p={pval:.3e}")
#
#     # 存储结果
#     stats_results.append({
#         'gene': gene,
#         'stat_kruskal': stat,
#         'pval_kruskal': pval,
#         'n_OS1_low': len(group_low),
#         'n_OS1_high': len(group_high),
#         'n_OS0_keep': len(group_os0)
#     })
#
# # 隐藏多余子图
# for j in range(len(selected_genes), nrows * ncols):
#     axes[j].axis('off')
#
# plt.tight_layout()
# with PdfPages(violin_pdf_path) as pdf:
#     pdf.savefig(fig)
# plt.close(fig)
# print(f"三组小提琴图已保存到: {violin_pdf_path}")
#
# # 将统计检验结果保存
# results_df = pd.DataFrame(stats_results)
# results_path = os.path.join(save_path, "three_group_stats_results.csv")
# results_df.to_csv(results_path, index=False)
# print(f"统计检验结果已保存到: {results_path}")
