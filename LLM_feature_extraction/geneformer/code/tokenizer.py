"""
Geneformer tokenizer.

Input data:
Required format: raw counts scRNAseq data without feature selection as .loom file
Required row (gene) attribute: "ensembl_id"; Ensembl ID for each gene
Required col (cell) attribute: "n_counts"; total read counts in that cell
Optional col (cell) attribute: "filter_pass"; binary indicator of whether cell should be tokenized based on user-defined filtering criteria
Optional col (cell) attributes: any other cell metadata can be passed on to the tokenized dataset as a custom attribute dictionary as shown below

Usage:
  from geneformer import TranscriptomeTokenizer
  tk = TranscriptomeTokenizer({"cell_type": "cell_type", "organ_major": "organ_major"}, nproc=4)
  tk.tokenize_data("loom_data_directory", "output_directory", "output_prefix")
"""

from __future__ import annotations
from typing import Literal
import pickle
from pathlib import Path
from tqdm import tqdm
import logging

import warnings
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import anndata as ad
import loompy as lp
import numpy as np
import scipy.sparse as sp
from datasets import Dataset

logger = logging.getLogger(__name__)

GENE_MEDIAN_FILE = Path(__file__).parent / "gene_median_dictionary.pkl"
TOKEN_DICTIONARY_FILE = Path(__file__).parent / "token_dictionary.pkl"


def rank_genes(gene_vector, gene_tokens):
    """
    Rank gene expression vector.
    """
    # sort by median-scaled gene values
    sorted_indices = np.argsort(-gene_vector)
    return gene_tokens[sorted_indices]


def tokenize_cell(gene_vector, gene_tokens):
    """
    Convert normalized gene expression vector to tokenized rank value encoding.
    """
    # create array of gene vector with token indices
    # mask undetected genes
    nonzero_mask = np.nonzero(gene_vector)[0]
    # rank by median-scaled gene values
    return rank_genes(gene_vector[nonzero_mask], gene_tokens[nonzero_mask])


class TranscriptomeTokenizer:
    def __init__(
        self,
        custom_attr_name_dict=None,
        nproc=1,
        gene_median_file=GENE_MEDIAN_FILE,
        token_dictionary_file=TOKEN_DICTIONARY_FILE,
        norm_read=True,
        trunc_num=2048
    ):
        """
        Initialize tokenizer.

        Parameters
        ----------
        custom_attr_name_dict : None, dict
            Dictionary of custom attributes to be added to the dataset.
            Keys are the names of the attributes in the loom file.
            Values are the names of the attributes in the dataset.
        nproc : int
            Number of processes to use for dataset mapping.
        gene_median_file : Path
            Path to pickle file containing dictionary of non-zero median
            gene expression values across Genecorpus-30M.
        token_dictionary_file : Path
            Path to pickle file containing token dictionary (Ensembl IDs:token).
        """
        # dictionary of custom attributes {output dataset column name: input .loom column name}
        self.custom_attr_name_dict = custom_attr_name_dict

        # number of processes for dataset mapping
        self.nproc = nproc

        # load dictionary of gene normalization factors
        # (non-zero median value of expression across Genecorpus-30M)
        with open(gene_median_file, "rb") as f:
            self.gene_median_dict = pickle.load(f)

        # load token dictionary (Ensembl IDs:token)
        with open(token_dictionary_file, "rb") as f:
            self.gene_token_dict = pickle.load(f)

        # gene keys for full vocabulary
        self.gene_keys = list(self.gene_median_dict.keys())

        # protein-coding and miRNA gene list dictionary for selecting .loom rows for tokenization
        self.genelist_dict = dict(zip(self.gene_keys, [True] * len(self.gene_keys)))
        ## should load data to memory
        self.norm_read =  norm_read
        self.trunc_num = trunc_num
    def tokenize_data(
        self,
        data_directory: Path | str,
        output_directory: Path | str,
        output_prefix: str,
        file_format: Literal["loom", "h5ad"] = "loom",
        use_generator: bool = False,
    ):
        """
        Tokenize .loom files in loom_data_directory and save as tokenized .dataset in output_directory.

        Parameters
        ----------
        loom_data_directory : Path
            Path to directory containing loom files or anndata files
        output_directory : Path
            Path to directory where tokenized data will be saved as .dataset
        output_prefix : str
            Prefix for output .dataset
        file_format : str
            Format of input files. Can be "loom" or "h5ad".
        use_generator : bool
            Whether to use generator or dict for tokenization.
        """
        tokenized_cells, cell_metadata = self.tokenize_files(
            Path(data_directory), file_format
        )
        tokenized_dataset = self.create_dataset(tokenized_cells, cell_metadata, use_generator=use_generator,trunc_num = self.trunc_num)

        output_path = (Path(output_directory) / output_prefix).with_suffix(".dataset")
        tokenized_dataset.save_to_disk(output_path)
        return tokenized_dataset,tokenized_cells, cell_metadata

    def tokenize_data_mid(
            self,
            data_directory: Path | str,
            file_format: Literal["loom", "h5ad"] = "loom",
    ):
        """
        Tokenize files in data_directory and return tokenized cells and cell metadata.

        Parameters
        ----------
        data_directory : Path
            Path to directory containing loom files or anndata files
        file_format : str
            Format of input files. Can be "loom" or "h5ad".

        Returns
        -------
        tokenized_cells, cell_metadata
        """
        tokenized_cells, cell_metadata = self.tokenize_files(Path(data_directory), file_format)
        return tokenized_cells, cell_metadata

    def create_and_save_dataset(
            self,
            tokenized_cells,
            cell_metadata,
            output_directory: Path | str,
            output_prefix: str,
            use_generator: bool = False,
    ):
        """
        Create a dataset from tokenized cells and cell metadata, and save it to disk.

        Parameters
        ----------
        tokenized_cells : list
            Tokenized cells from the input data
        cell_metadata : dict
            Metadata associated with the tokenized cells
        output_directory : Path
            Path to directory where tokenized data will be saved as .dataset
        output_prefix : str
            Prefix for output .dataset
        use_generator : bool
            Whether to use generator or dict for tokenization.
        """
        tokenized_dataset = self.create_dataset(tokenized_cells, cell_metadata, use_generator=use_generator,trunc_num = self.trunc_num)

        output_path = (Path(output_directory) / output_prefix).with_suffix(".dataset")
        tokenized_dataset.save_to_disk(output_path)
        return tokenized_dataset



    def tokenize_files(
        self, data_directory, file_format: Literal["loom", "h5ad"] = "loom"
    ):
        tokenized_cells = []
        if self.custom_attr_name_dict is not None:
            cell_attr = [attr_key for attr_key in self.custom_attr_name_dict.keys()]
            cell_metadata = {attr_key: [] for attr_key in self.custom_attr_name_dict.values()}

        # loops through directories to tokenize .loom files
        file_found = 0
        # loops through directories to tokenize .loom or .h5ad files
        tokenize_file_fn = (
            self.tokenize_loom if file_format == "loom" else self.tokenize_anndata
        )
        for file_path in data_directory.glob("*.{}".format(file_format)):
            file_found = 1
            print(f"Tokenizing {file_path}")
            file_tokenized_cells, file_cell_metadata = tokenize_file_fn(file_path)
            tokenized_cells += file_tokenized_cells
            if self.custom_attr_name_dict is not None:
                for k in cell_attr:
                    cell_metadata[self.custom_attr_name_dict[k]] += file_cell_metadata[k]
            else:
                cell_metadata = None

        if file_found == 0:
            logger.error(
                f"No .{file_format} files found in directory {data_directory}.")
            raise
        return tokenized_cells, cell_metadata

    def tokenize_anndata(self, adata_file_path, target_sum=10_000, chunk_size=512):
        print('anndata mode')

        if self.norm_read:
            # 将数据从磁盘加载到内存中，避免 .copy() 和视图嵌套问题
            adata = ad.read(adata_file_path, backed="r").to_memory()
        else:
            adata = ad.read(adata_file_path, backed="r")

        # 打印 AnnData 对象的维度 (细胞数, 基因数)
        print(f"adata shape: {adata.shape}")

        # 打印前几行的 obs 数据（细胞的元数据信息）
        print("adata.obs head (first few rows of cell metadata):")
        print(adata.obs.head())

        # 打印前几行的 var 数据（基因的元数据信息）
        print("adata.var head (first few rows of gene metadata):")
        print(adata.var.head())

        # 打印 X 矩阵的数据结构（基因表达数据）
        print("adata.X data structure:")
        print(f"adata.X type: {type(adata.X)}")
        print(f"adata.X shape: {adata.X.shape}")

        # 打印 X 矩阵的行名（细胞名）
        print("adata.X row (cell) names:")
        print("head: ",adata.obs_names[:5])  # 打印前5个细胞名称
        print("last: ",adata.obs_names[-5:])  # 打印后5个细胞名称
        # 打印 X 矩阵的列名（基因名）
        print("adata.X column (gene) names:")
        print("head: ",adata.var_names[:5])  # 打印前5个基因名称
        print("last: ",adata.var_names[-5:])  # 打印后5个基因名称
        # 打印 X 矩阵的前几行数据
        print("adata.X head (first few rows of expression data):")
        print(adata.X[:5, :5].toarray() if isinstance(adata.X, sp.spmatrix) else adata.X[:5, :5])

        # 如果有自定义属性，初始化元数据字典
        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        # 获取 protein-coding 和 miRNA 基因的索引
        coding_miRNA_loc = np.where(
            [self.genelist_dict.get(i, False) for i in adata.var["ensembl_id"]]
        )[0]
        norm_factor_vector = np.array(
            [
                self.gene_median_dict[i]
                for i in adata.var["ensembl_id"][coding_miRNA_loc]
            ]
        )
        coding_miRNA_ids = adata.var["ensembl_id"][coding_miRNA_loc]
        coding_miRNA_tokens = np.array(
            [self.gene_token_dict[i] for i in coding_miRNA_ids]
        )

        # 检查是否有 filter_pass 列
        try:
            _ = adata.obs["filter_pass"]
        except KeyError:
            var_exists = False
        else:
            var_exists = True

        if var_exists:
            filter_pass_loc = np.where(
                [i == 1 for i in adata.obs["filter_pass"]]
            )[0]
        elif not var_exists:
            print(
                f"{adata_file_path} has no column attribute 'filter_pass'; tokenizing all cells."
            )
            # filter_pass_loc = np.array([i for i in range(adata.shape[0])])
            filter_pass_loc = np.arange(adata.shape[0])

        tokenized_cells = []

        print('start mapping')

        for i in tqdm(range(0, len(filter_pass_loc), chunk_size), desc="Tokenizing cells from AnnData"):
            idx = filter_pass_loc[i:i+chunk_size]

            n_counts = adata[idx].obs['n_counts'].values[:, None]
            X_view = adata[idx, coding_miRNA_loc].X
            # # 第一步：按行索引并复制，避免生成视图
            # adata_sub = adata[idx, :].copy()
            #
            # # 第二步：再按列索引
            # X_view = adata_sub[:, coding_miRNA_loc].X

            X_norm = (X_view / n_counts * target_sum / norm_factor_vector)
            X_norm = sp.csr_matrix(X_norm)

            tokenized_cells += [
                rank_genes(X_norm[i].data, coding_miRNA_tokens[X_norm[i].indices])
                for i in range(X_norm.shape[0])
            ]

            # add custom attributes for subview to dict
            if self.custom_attr_name_dict is not None:
                for k in file_cell_metadata.keys():
                    file_cell_metadata[k] += adata[idx].obs[k].tolist()
            else:
                file_cell_metadata = None

        return tokenized_cells, file_cell_metadata

    def tokenize_loom(self, loom_file_path, target_sum=10_000):
        print('loom mode')
        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        with lp.connect(str(loom_file_path)) as data:
            # define coordinates of detected protein-coding or miRNA genes and vector of their normalization factors
            coding_miRNA_loc = np.where(
                [self.genelist_dict.get(i, False) for i in data.ra["ensembl_id"]]
            )[0]
            norm_factor_vector = np.array(
                [
                    self.gene_median_dict[i]
                    for i in data.ra["ensembl_id"][coding_miRNA_loc]
                ]
            )
            coding_miRNA_ids = data.ra["ensembl_id"][coding_miRNA_loc]
            coding_miRNA_tokens = np.array(
                [self.gene_token_dict[i] for i in coding_miRNA_ids]
            )

            # define coordinates of cells passing filters for inclusion (e.g. QC)
            try:
                data.ca["filter_pass"]
            except AttributeError:
                var_exists = False
            else:
                var_exists = True

            if var_exists:
                filter_pass_loc = np.where(
                    [i == 1 for i in data.ca["filter_pass"]]
                )[0]
            elif not var_exists:
                print(
                    f"{loom_file_path} has no column attribute 'filter_pass'; tokenizing all cells."
                )
                # filter_pass_loc = np.array([i for i in range(data.shape[1])])
                filter_pass_loc = np.arange(data.shape[1])
            # scan through .loom files and tokenize cells
            tokenized_cells = []
            print('start mapping')

            for (_ix, _selection, view) in tqdm(data.scan(items=filter_pass_loc, axis=1), desc="Tokenizing cells from Loom"):
                # select subview with protein-coding and miRNA genes
                subview = view.view[coding_miRNA_loc, :]

                # normalize by total counts per cell and multiply by 10,000 to allocate bits to precision
                # and normalize by gene normalization factors
                subview_norm_array = (
                    subview[:, :]
                    / subview.ca.n_counts
                    * target_sum
                    / norm_factor_vector[:, None]
                )
                # tokenize subview gene vectors
                tokenized_cells += [
                    tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens)
                    for i in range(subview_norm_array.shape[1])
                ]

                # add custom attributes for subview to dict
                if self.custom_attr_name_dict is not None:
                    for k in file_cell_metadata.keys():
                        file_cell_metadata[k] += subview.ca[k].tolist()
                else:
                    file_cell_metadata = None

        return tokenized_cells, file_cell_metadata

    def create_dataset(self, tokenized_cells, cell_metadata, use_generator=False, trunc_num = 2048):
        print("Creating dataset.")
        # create dict for dataset creation
        dataset_dict = {"input_ids": tokenized_cells}
        if self.custom_attr_name_dict is not None:
            dataset_dict.update(cell_metadata)

        # create dataset
        if use_generator:
            def dict_generator():
                for i in range(len(tokenized_cells)):
                    yield {k: dataset_dict[k][i] for k in dataset_dict.keys()}
            output_dataset = Dataset.from_generator(dict_generator, num_proc=self.nproc)
        else:
            output_dataset = Dataset.from_dict(dataset_dict)

        # # truncate dataset
        # def truncate(example):
        #     example["input_ids"] = example["input_ids"][:2048]
        #     return example

        def truncate(batch):
            batch["input_ids"] = [x[:trunc_num] for x in batch["input_ids"]]
            #batch["input_ids"] = [x[:5000] for x in batch["input_ids"]]
            return batch

        output_dataset_truncated = output_dataset.map(truncate, batched=True, batch_size=100, num_proc=self.nproc)

        # # measure lengths of dataset
        # def measure_length(example):
        #     example["length"] = len(example["input_ids"])
        #     return example

        # 修改 measure_length 函数
        def measure_length(batch):
            batch["length"] = [len(x) for x in batch["input_ids"]]
            return batch

        output_dataset_truncated_w_length = output_dataset_truncated.map(
            measure_length, batched=True, batch_size=100,num_proc=self.nproc
        )

        return output_dataset_truncated_w_length












