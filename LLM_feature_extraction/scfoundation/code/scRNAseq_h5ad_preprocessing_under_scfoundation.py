
import os
import sys
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="VAE Pretraining")
def strict_str2bool(v):
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Only "True" or "False" are allowed.')

parser.add_argument('--system_path', type=str, default='/home/luozeyu/desktop/scFoundation-main/preprocessing',
                    help=' ')
parser.add_argument('--file_name', type=str, default='panglao_human_1_10.h5ad',
                    help=' ')
parser.add_argument('--sparse_matrix', type=strict_str2bool, default=True, help='Specify whether to use a sparse matrix (True) or not (False).')

args = parser.parse_args()

target_path = args.system_path
file_name=args.file_name
sparse_matrix = args.sparse_matrix
print(sparse_matrix)

sys.path.append(target_path)

os.chdir(target_path)

from scRNA_workflow import *


sc.settings.figdir='./figures_new/' # set figure folder

path = f"./data/{file_name}"

save_path = './output'

# read from csv file
adata = sc.read_h5ad(path)
if sparse_matrix:
    X_df= pd.DataFrame(sparse.csr_matrix.toarray(adata.X),index=adata.obs.index.tolist(),columns=adata.var.index.tolist()) # read from csv file
else:
    X_df = pd.DataFrame(adata.X, index=adata.obs.index.tolist(),
                        columns=adata.var.index.tolist())  # read from csv file
gene_list_df = pd.read_csv('./OS_scRNA_gene_index.19264.tsv', header=0, delimiter='\t')
gene_list = list(gene_list_df['gene_name'])
X_df, to_fill_columns, var = main_gene_selection(X_df, gene_list)
adata_uni = sc.AnnData(X_df)
adata_uni.obs = adata.obs
adata_uni.uns = adata.uns

adata_uni = BasicFilter(adata_uni,qc_min_genes=200,qc_min_cells=0) # filter cell and gene by lower limit
adata_uni = QC_Metrics_info(adata_uni)


if not os.path.exists(save_path):
    os.makedirs(save_path)

save_adata_h5ad(adata_uni,save_path+f"/preprocessed_all_{file_name}")



###save filter cell info

cell_ids = adata_uni.obs.index
print(cell_ids)

try:
    batch_data = adata_uni.obs['Batch']
    df = pd.DataFrame({
        'Cell_ID': cell_ids,
        'Batch': batch_data
    })
except KeyError:
    df = pd.DataFrame({
        'Cell_ID': cell_ids
    })
    print("The 'Batch' column does not exist in adata_uni.obs.")


file_name_dex = file_name.replace('.h5ad', '_info.xlsx')
df.to_excel(os.path.join(save_path, f'preprocessed_all_{file_name_dex}'), index=False)
