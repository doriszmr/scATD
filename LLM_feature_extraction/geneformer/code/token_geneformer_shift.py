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


parser = argparse.ArgumentParser(description="Set up and use command-line arguments for model handling.")

parser.add_argument("--open_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/in/your_data.h5ad', help="Path to open input data.")
parser.add_argument("--save_path", type=str, default='/home/luozeyu/desktop/VAE_pretraining/output/VAE_inference', help="Path to save output data.")
parser.add_argument("--open_path_conference_data", type=str, default='/home/luozeyu/desktop/VAE_pretraining/data/conference_data', help="Path to open conference data.")
parser.add_argument("--mapping_file", type=str, help="Filename of the gene mapping file to load.")
parser.add_argument("--G38_convert", type=strict_str2bool, default=False, help="if G38 need, defult only result in G37.")

args = parser.parse_args()

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


mapping_path = os.path.join(args.open_path_conference_data, args.mapping_file)
gene_symbol_to_ensembl = load_gene_symbol_to_ensembl_mapping(mapping_path)


file_name = os.listdir(open_path)[0]

input_file_name = os.path.splitext(file_name)[0]


output_file_path = os.path.join(save_path, f"{input_file_name}_n_counts.h5ad")

adata = sc.read_h5ad(os.path.join(open_path,file_name))

if isinstance(adata.X, np.ndarray):
    adata.obs["n_counts"] = np.sum(adata.X, axis=1)
else:
    adata.obs["n_counts"] = np.array(adata.X.sum(axis=1)).flatten()


print(adata.obs["n_counts"])

def convert_gene_names_to_ensembl(adata, gene_symbol_to_ensembl, none_name_save, match_name_save):
    """
    Function: Convert gene names in adata.var_names to Ensembl IDs and store them in adata.var['ensembl_id'].
    
    Parameters:
    
    adata: An AnnData object.
    gene_symbol_to_ensembl: A dictionary that maps gene symbols to Ensembl IDs.
    none_name_save: The filename to save genes that could not be mapped.
    match_name_save: The filename to save genes that were successfully mapped.
    Returns:
    
    adata: The updated AnnData object.
    modified_gene_names: A list of modified gene names.
    ENSG_name: A list of Ensembl IDs.
    """

    print(f"raw adata dim {adata.shape}")

    print("the first five gene name", adata.var_names[:5])

    ensembl_ids = []
    for gene_name in adata.var_names:
        # if gene name is 'ENSxxxxx_GeneSymbol', extract Ensembl ID
        if '_' in gene_name and gene_name.startswith('ENS'):
            ensembl_id = gene_name.split('_')[0]
            ensembl_ids.append(ensembl_id)

        # if gene name is Ensembl ID
        elif gene_name.startswith('ENS'):
            ensembl_ids.append(gene_name)

        # If the gene name is the gene symbol (like ASD1), try to obtain the Ensembl ID through the mapping dictionary.
        else:
            ensembl_id = gene_symbol_to_ensembl.get(gene_name)
            if ensembl_id:
                ensembl_ids.append(ensembl_id)
            else:
                # if can not mapping，mark it as None
                ensembl_ids.append(None)


    adata.var['ensembl_id'] = ensembl_ids

    none_mask = adata.var['ensembl_id'].isnull()
    none_entries = adata.var[none_mask]
    none_entries.to_excel(none_name_save, index=True)
    print(f"The genes that could not be mapped have been saved to: {none_name_save}")

    num_none_genes = none_entries.shape[0]
    print(f"Number of genes that could not be mapped: {num_none_genes}")


    adata = adata[:, ~none_mask.values]

    num_genes_after_filter = adata.shape[1]
    num_genes_before_filter = len(ensembl_ids)
    num_filtered_out = num_genes_before_filter - num_genes_after_filter
    print(f"Number of original genes: {num_genes_before_filter}, Number of filtered genes: {num_genes_after_filter}")
    print(f"The number of genes filtered: {num_filtered_out},is equal to the number of genes that could not be mapped: {num_filtered_out == num_none_genes}")

    modified_gene_names = adata.var_names.tolist()
    ENSG_name = adata.var['ensembl_id']

    combined_df = pd.DataFrame({
        'Gene_Name': modified_gene_names,
        'Ensembl_ID': ENSG_name
    })

    combined_df.to_excel(match_name_save, index=False)
    print(f"The genes that were successfully mapped have been saved to {match_name_save}")

    return adata, modified_gene_names, ENSG_name





none_name_save = os.path.join(save_path, f"{input_file_name}_none_ensembl_ids.xlsx")

match_name_save = os.path.join(save_path, f"{input_file_name}_match_ensembl_ids.xlsx")


adata_convert, modified_gene_names,ENSG_name = convert_gene_names_to_ensembl(adata, gene_symbol_to_ensembl, none_name_save,match_name_save)


modified_genes_file = os.path.join(save_path, f"{input_file_name}_modified_gene_names.txt")
with open(modified_genes_file, 'w') as f:
    for gene_name in modified_gene_names:
        f.write(f"{gene_name}\n")
print(f"The modified gene names have been saved to {modified_genes_file}")


ENSG_name_file = os.path.join(save_path, f"{input_file_name}_ENSG_name_gene_names.txt")
with open(ENSG_name_file, 'w') as f:
    for gene_name in ENSG_name:
        f.write(f"{gene_name}\n")
print(f"The modified gene names have been saved to {ENSG_name_file}")


adata_convert.write(output_file_path)

print(f"data saved at {output_file_path}")


