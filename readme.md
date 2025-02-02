# scATD: A De Novo Adaptive Transfer and Distillation Framework Based on LLM for single-cell Drug Sensitivity Prediction and Biomarker Identification

## Introduction

### Author Contact Information:

- Author 1: Zeyu Luo, Email: [1024226968@qq.com](mailto:1024226968@qq.com), ORCID: 0000-0001-6650-9975
- Author 2: Murong Zhou , Email: doris_zmr@sina.com
- Author 3: Yu-Hang Yin, Email: zjyinyh@gmail.com

Your contributions, feedback, and suggestions are highly appreciated. If you encounter any issues or have questions, feel free to reach out to the authors via the provided email addresses. Thank you for your interest in our work!

## Dataset Available

All Data can be download in figshare (https://figshare.com/articles/software/scATD/27908847).
Additionally, the model checkpoints for VAE_sf, VAE_gf, and Dist_VAE, which were pretrained on Panglao data, are also available for download on figshare (https://figshare.com/articles/software/scATD/27908847).
Finally, the transfer learning and drug resistance models can be trained by following the transfer learning instructions and code provided, and its backbone is the above pretrained VAE model.
## scATD Environment Setup

For the environment configuration for feature extraction using scFoundation or geneformer, please refer to the configuration in the 'Feature Extraction from LLM' section.

For the environment configuration of other scATD modules

1. Clone this repository or download the project files.
2. Navigate to the project directory.
3. Create a new Conda environment with Python version `>= 3.9`, then activate the environment:

```bash
conda create -n scATD-env python=3.9
conda activate scATD-env
```

  4.For CPU-only setup (if you don't need GPU acceleration):

```bash
pip install .
```

  5.For CUDA Support (GPU Acceleration), install the additional dependencies required for GPU acceleration:

```bash
pip install .[cuda]
```

For more detailed installation instructions and troubleshooting, please refer to the PyTorch website, which provides guides for setting up CUDA on different systems.



## Data Preprocessing and Feature Extraction from LLM 

### scFoundation 

**Step 0:** Please follow the official open-source repository of scFoundation on GitHub (https://github.com/biomap-research/scFoundation/tree/main/model) to download the model source code and set up the relevant environment. Since this model adheres to the Apache License 2.0, in order to respect the original author's copyright, we only provide the methods and command scripts to run the model below and provide a copy of the original [**scFoundation**](https://github.com/biomap-research/scFoundation/tree/main/model) project, including all model files, as a compressed package within our project `.\LLM_feature_extraction\scfoundation\original_scfoundation_project`. We have not made any changes to the original author's Python scripts.

**Step 1: Preprocessing RNA-seq Gene Expression Data** Download the RNA-seq gene expression dataset in h5ad format containing unnormalized raw counts. Process the data into a matrix with rows representing cells and columns representing genes (using Gene Symbols).

**Step 2:** In the path `.\scFoundation-main\preprocessing\`, create new folders `\code` and `\data`. Then, in the folder `.\scFoundation-main\preprocessing\code`, create the script `scRNAseq_h5ad_preprocessing_under_scfoundation.py`and `scRNA_workflow.py`. These scripts are located in `.\LLM_feature_extraction\scfoundation\code` within this project. Also, place the data from Step 1 into `.\scFoundation\preprocessing\data`.

Then conduct code below (e.g., `ALL_Bulk_Dell.h5ad as example`).

```bash
python ./scFoundation-main/preprocessing/code/scRNAseq_h5ad_preprocessing_under_scfoundation.py --system_path ./scFoundation-main/preprocessing --file_name ALL_Bulk_Dell.h5ad --sparse_matrix False
```

**Explanation:**

- The `.\` at the beginning represents the path to the scFoundation project scripts downloaded to your local machine or server.
- The preprocessed files will ultimately be output to `.\scFoundation-main\preprocessing\output`.

------

**Step 3: Extracting Cell Embeddings (Feature Matrix) for RNA-seq Data**

First, in the folder `.\scFoundation-main\model\examples`, create a new folder named `single_cell_data`. Place the preprocessed data generated in Step 2 (e.g., `preprocessed_ALL_Bulk_Dell.h5ad`) into `.\scFoundation-main\model\examples\single_cell_data`. Then, create `.\scFoundation-main\model\output\single_cell_data` to save the feature matrix. You also need follow the description file in `scFoundation-main\model\models` to download scfoundation model. 

You can refer to the following file path arrangement and execute the feature extraction code `get_embedding.py` according to different data types.   

```bash
scFoundation-main
│
├── model
    │
    ├── get_embedding.py     
    └── examples
    │   │
    │   └── single_cell_data         
    │   │   │       
    │       └── <you can put your data here>                         
    └── output
        │
        └── single_cell_data
            │
            └── <output files here>  
```

For bulk feature extraction (preprocessed_all_ALL_Bulk_Dell.h5ad as example)

```bash
python get_embedding.py --task_name AllbulkDEll --input_type bulk --output_type cell --pool_type all --data_path ./examples/single_cell_data/preprocessed_all_ALL_Bulk_Dell.h5ad --pre_normalized F --version rde --save_path ./output/single_cell_data --tgthighres f1
```

For single cell feature extraction (preprocessed_GSE149214.h5ad as example)

```bash
python get_embedding.py --task_name FA34 --input_type singlecell --output_type cell --pool_type all --tgthighres a5 --data_path ./examples/single_cell_data/preprocessed_GSE149214.h5ad --pre_normalized F --version rde --save_path ./output/single_cell_data --tgthighres f1
```

More information please reference to https://github.com/biomap-research/scFoundation/tree/main/model. 



### geneformer

**Step 0:** Please follow the official open-source repository of geneformer on GitHub ([jkobject/geneformer](https://github.com/jkobject/geneformer)) to download the model source code and set up the relevant environment.  In order to respect the original author's copyright, we provide only the methods and command scripts necessary to run the model, along with essential changes to the model's execution scripts. We clearly indicate the modified scripts in the following sections. We also include a copy of the original [**Geneformer**](https://github.com/jkobject/geneformer) project, encompassing all model files, as a compressed package within our project  `.\LLM_feature_extraction\geneformer\original_geneformer_project`. No modifications have been made to any files in this copy version. We provide instructions for replacing certain raw code in Geneformer as outlined below. **Note:** After downloading or extracting `geneformer-main`, you need to replace this code **before** using `pip install` to set up the Geneformer environment.

The  path for raw script code storage is located in the `geneformer-main` directory:

`geneformer-main\geneformer\tokenizer.py` 

`geneformer-main\geneformer\in_silico_perturber.py` 

`geneformer-main\geneformer\emb_extractor.py`

 Replace each with corresponding code  storage in `.\LLM_feature_extraction\geneformer\code`



**Step 1: Preprocessing RNA-seq Gene Expression Data** Download the RNA-seq gene expression dataset in h5ad format containing unnormalized raw counts. Process the data into a matrix with rows representing cells and columns representing genes (using Ensembl ID). Besides considering that the GDSC database stores gene names in the **Gene Symbol** format, we have developed a gene type conversion tool to facilitate the transformation of gene name from **Gene Symbols** to **Ensembl IDs**. You can set environment and running code below (`token_geneformer_shift.py` storage in `.\LLM_feature_extraction\geneformer\code` ,and refer below path parameter creating your path. Besides `mart_export.txt` is the gene shift file storage in  `.\LLM_feature_extraction\geneformer\preprocess\data\conference_data`).  

```bash
python ./geneformer-main/preprocess/code/token_geneformer_shift.py --open_path ./geneformer-main/preprocess/data/in/ --save_path ./geneformer-main/preprocess/output --open_path_conference_data ./geneformer-main/preprocess/data/conference_data/ --mapping_file mart_export.txt 
```

`--open_path:` the path of data need gene ID shift , We use `GSE112274.h5ad` as an example, which can be found in `.\LLM_feature_extraction\geneformer\data_example`.

`--save_path:` the path of gene ID shift outcome  data. The output H5AD file is the result after the gene ID conversion (e.g., `GSE112274_n_counts.h5ad`), while other files contain information such as gene filtering details.

`--open_path_conference_data: `the path of `mart_export.txt`located. 

**Step 2:** Extracting Cell Embeddings (Feature Matrix) for RNA-seq Data

For  gf-6L-30M-original version，you can running script as below，`Embedding.py` is storage in `.\LLM_feature_extraction\geneformer\code`

#### gf-6L-30M-original

```bash
python ./geneformer-main/Embedding/code/Embedding.py --open_path ./geneformer-main/Embedding/data/in/ --save_path ./geneformer-main/Embedding/output/tokenized_dataset/ --save_path_embedding ./geneformer-main/Embedding/output/feature_embedding/6L --open_path_conference_data ./geneformer-main/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224 --device_choose cuda:0  --re_load False --trunc_num 2048 --batch_size 32
```

`--open_path:` the path of data need feature extraction, 

`--save_path:` the outcome of tokenized file

`--save_path_embedding:` the outcome of feature extraction

`--open_path_conference_data:` the path of storage gf-6L-30M-original model, the default set `./geneformer-main/fine_tuned_models/geneformer-6L-30M_CellClassifier_cardiomyopathies_220224` is the same path for `geneformer-main` gf-6L-30M-original model storage.

For  gf-12L-30M-i2048 version，you can running script as below，`Embedding.py` is storage in `.\LLM_feature_extraction\geneformer\code`. Besides, you need follow the readme.txt located in our project `.\LLM_feature_extraction\geneformer\gf-12L-30M-i2048` to download gf-12L-30M-i2048 model from figshare.

#### gf-12L-30M-i2048

```bash
python ./geneformer-main/Embedding/code/Embedding.py --open_path ./geneformer-main/Embedding/data/in/ --save_path ./geneformer-main/Embedding/output/tokenized_dataset/12L --save_path_embedding ./geneformer-main/Embedding/output/feature_embedding/ --open_path_conference_data ./geneformer-main/Embedding/data/conference_data/gf-12L-30M-i2048/ --device_choose cuda:0 --re_load False --trunc_num 2048 --batch_size 32
```

`--open_path:` the path of data need feature extraction, 

`--save_path:` the outcome of tokenized file

`--save_path_embedding:` the outcome of feature extraction

`--open_path_conference_data:` the path of storage gf-12L-30M-i2048 model, the model file are storaged in `.\LLM_feature_extraction\geneformer\gf-12L-30M-i2048`, which is download from Hugging Face [ctheodoris/Geneformer at main](https://huggingface.co/ctheodoris/Geneformer/tree/main).

More information please reference to [jkobject/geneformer](https://github.com/jkobject/geneformer)

### Dist_VAE/VAE_sf/VAE_gf

**Data preprocessing instruction:** Please follow **Steps 1** and **2** of the scFoundation data preprocessing methods, using the preprocessed `RNA-seq.h5ad` data as input for the Dist_VAE. When preparing the `RNA-seq.h5ad` data, adhere to the instructions in the section below to perform Dist_VAE model training and prediction. 

Additionally, both VAE_sf and VAE_gf fundamentally accept feature matrices extraction from scFoundation or GeneFormer as input. Therefore, their preprocessing of raw data is contingent upon the methodologies used in scFoundation or GeneFormer, eliminating the need for an independent raw data processing procedure.



## Res_VAE Pretraining (pretraining in Panglao data)

### VAE_sf hyperparameter optimization and Pretraining
Step1 VAE_sf hyperparameter optimization

1. **Prepare Input Data**
   -  Place the scFoundation model-derived feature data (`.npy` files) generated from previous steps into the **user-specified directory**:

2. **Run Hyperparameter Optimization**
Execute the following script to train the model and output the optimal hyperparameter configuration file (best_hyperparameters.xlsx):

```bash
python ./Res_VAE_pretraining/skf_pretraining/code/VAE_sf_Res-VAE_hyperparam_pretraining.py 
```
    --open_path ./data/ \                     # Path to input features (.npy),Parameter passing is unsupported; please modify directly in the script.
    --save_path_outer ./Res_VAE_retraining_after_hyperparameter/output \  # Output directory,Parameter passing is unsupported; please modify directly in the script.
    --file_prefix scRNA-seq_panglao \  # File naming prefix,Parameter passing is unsupported; please modify directly in the script.

After executing the above code, users can obtain the optimal hyperparameter configuration file. Additionally, we recommend that users directly use the optimal hyperparameter configuration file we provide.

Step2 VAE_sf Pretraining after hyperparameter optimization

1. **Prepare Input Data**
   -  Place the scFoundation model-derived feature data (`.npy` files) generated from previous steps into the **user-specified directory**:

2. **Run 10-fold cross-validation for model pretraining**
Execute the following script to train the model and output the model checkpoint file (default is using the last epoch fold 1 checkponit as final model for downstream task, we have also provided pre-trained model checkpoints on Figshare):

```bash
python ./Res_VAE_pretraining/pretraining_after_hyperparameter/code/VAE_sf_Res-VAEpretraining.py \
    --open_path ./data/ \                     # Path to input features (.npy)
    --save_path_outer ./Res_VAE_retraining_after_hyperparameter/output \  # Output directory
    --open_path_conference_data ./data/conference_data \  # hyperparameter file path, user should put hyperparameter file here (Step1 output), or refer to the example data.
    --file_prefix scRNA-seq_panglao \  # File naming prefix
    --epoch_start_for_loss_plot_only 1 \      # Start epoch for loss visualization
    --batch_size 128 \                        # Training batch size
    --REC_beta 1000 \                         # Reconstruction loss weight (β)
    --best_parameter_name VAE_sf_best_hyperparameters.xlsx        # hyperparameter file name
```


### VAE_gf hyperparameter optimization and Pretraining
For VAE_gf hyperparameter optimization and pretraining, you can follow a **similar procedure to the VAE-sf** described above.

Step1 VAE_gf hyperparameter optimization

```bash
python ./Res_VAE_pretraining/skf_pretraining/code/VAE_gf_Res-VAE_hyperparam_pretraining.py
```
    --open_path ./data/ \                     # Path to input features (.npy),Parameter passing is unsupported; please modify directly in the script.
    --save_path_outer ./Res_VAE_retraining_after_hyperparameter/output \  # Output directory,Parameter passing is unsupported; please modify directly in the script.
    --file_prefix scRNA-seq_panglao \  # File naming prefix,Parameter passing is unsupported; please modify directly in the script.

After executing the above code, users can obtain the optimal hyperparameter configuration file. Additionally, we recommend that users directly use the optimal hyperparameter configuration file we provide.

Step2 VAE_gf Pretraining after hyperparameter optimization

```bash
python ./Res_VAE_pretraining/pretraining_after_hyperparameter/code/VAE_gf_Res-VAEpretraining.py \
    --open_path ./data/ \                     # Path to input features (.npy)
    --save_path_outer ./Res_VAE_retraining_after_hyperparameter/output \  # Output directory
    --open_path_conference_data ./data/conference_data \  # hyperparameter file path, user should put hyperparameter file here (Step1 output), or refer to the example data.
    --file_prefix scRNA-seq_panglao \  # File naming prefix
    --epoch_start_for_loss_plot_only 1 \      # Start epoch for loss visualization
    --batch_size 128 \                        # Training batch size
    --REC_beta 10000 \                         # Reconstruction loss weight (β)
    --best_parameter_name VAE_gf_best_hyperparameters.xlsx        # hyperparameter file name
```
Besides, default is using the last epoch fold 1 checkponit as final model for downstream task, we have also provided pre-trained model checkpoints on Figshare

## Distillation VAE Pretraining (pretraining in Panglao data)

```
We are currently undergoing peer review, so the code related to this part has not been made available. Please contact the author if needed.
```

## Transfer Learning and Model Inference 

### VAE_sf transfer learning and inference

```
We are currently undergoing peer review, so the code related to this part has not been made available. Please contact the author if needed.
```

### VAE_gf

```
We are currently undergoing peer review, so the code related to this part has not been made available. Please contact the author if needed.
```

### Dist_VAE

#### Domain adpative learning detail for Dist_VAE

```
We are currently undergoing peer review, so the code related to this part has not been made available. Please contact the author if needed.
```

#### Model evaluation and Inference detail for Dist_VAE

First, refer to the `Dist_VAE/inference` directory. we set `GSE140440` dataset as an example, you should place your evaluation dataset and labels (or the inference dataset) inside the `./Dist_VAE/inference/data/in/` directory.

Other path configurations should follow the parameter descriptions provided below. Once the paths and dependence model file (see below) are correctly set, you can execute the code to perform model evaluation or inference.

 **model inference and evalution running code**

```bash
python ./Dist_VAE/inference/code/Dist_VAE_inference.py --open_path ./Dist_VAE/inference/data/in/ --save_path ./Dist_VAE/inference/output/MMD_Dist_VAE/GSE140440  --model_configuration_path ./Dist_VAE/distillation_VAE_pretraining_model --Dist_VAE_DAL_model_path ./Dist_VAE/inference/data/Dist_VAE_DAL_model_path/GSE140440/ --model_inference_parameters_file checkpoint_fold5_final_epoch_150.pth --file_prefix GSE140440_dist_vae_infer  --batch_size 128 --device_choose cuda:6 --label_mapping "{\"sensitive\": 0, \"resistant\": 1}" --class_num 2 --drug_label_choose label --inference_only False
```

### 

`--open_path:`inference data input path 

`--save_path:` model prediction and evaluation matrix path

`--file_prefix:` output file prefix

`--batch_size:` inference batch number

`--device_choose:`  The argument `args.device_choose` allows the user to specify the device for model training or inference, where `"cpu"` forces the model to 				run on the CPU, `"cuda"` selects the default GPU (usually `"cuda:0"`), and `"cuda:<device_number>"` allows selection of a specific GPU by 				index (e.g., `"cuda:0"` for the first GPU, `"cuda:1"` for the second).

`--label_mapping:`convert class from word to number

`--model_configuration_path:` **Distillation VAE pretraining model path**, located at `./Dist_VAE/distillation_VAE_pretraining_model`. Please refer to the model.txt file in this directory for instructions download the model checkpoint from figshare.

`--Dist_VAE_DAL_model_path:` Dist_VAE inference model (after domain adaptive learning ) path. 

`--model_inference_parameters_file` **Dist_VAE inference model checkpoint**. Please refer to the model.txt file in this directory for instructions download the model checkpoint from figshare.

`--inference_only:` if only inference and not conduct evaluation. set True only conduct model inference (model deploy mode),set False conduct both model inference and evaluation (model evaluation mode). Note, when set False, `--label_mapping` must be specified, and a dataset with true labels (such as the example dataset) must be provided.



## Key Feature or Gene Identification 

```
We are currently undergoing peer review, so the code related to this part has not been made available. Please contact the author if needed.
```

## Comparison Experiments with Other Models

```
IF YOU ARE NOT THE REVIEWER:
We are currently undergoing peer review, so the experiment setting related to this part has not been made available. Please contact the author if needed.
IF YOU ARE THE REVIEWER:
Please see supplementary methods for details.
```


## Reference

Hao M, Gong J, Zeng X, et al. Large-scale foundation model on single-cell transcriptomics [J]. Nature Methods, 2024.

Theodoris C V, Xiao L, Chopra A, et al. Transfer learning enables predictions in network biology [J]. Nature, 2023, 618(7965): 616-24.

## Citation



## Acknowledgments

We are acknowledge the contributions of the open-source community and the developers of the Python libraries used in this study.

## Related Works
If you are interested in feature extraction and model interpretation for large language models, you may find our previous work helpful:
- **Interpretable feature extraction and dimensionality reduction in ESM2 for protein localization prediction**: [Link](https://academic.oup.com/bib/article/25/2/bbad534/7590319?login=false); [GitHub Repository](https://github.com/yujuan-zhang/feature-representation-for-LLMs)
