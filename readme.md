# scATD: A De Novo Adaptive Transfer and Distillation Framework Based on LLM for single-cell Drug Sensitivity Prediction and Biomarker Identification

## Introduction

### Author Contact Information:

- Author 1: Murong Zhou , Email: doris_zmr@sina.com
- Author 2: Zeyu Luo, Email: [1024226968@qq.com](mailto:1024226968@qq.com), ORCID: 0000-0001-6650-9975
- Author 3: Yu-Hang Yin, Email: zjyinyh@gmail.com

📌 scATD is a high-throughput and interpretable framework for single-cell drug response prediction (⚠️ in vitro cell experiment) based on pre-trained transcriptomic models combined with transfer learning. 

🔬 scATD incorporates three independently trained models: VAE-sf, VAE-gf, and Dist-VAE, based on different backbone feature extractors or knowledge distillation strategies.

🔴 **Note:** **VAE-sf corresponds to scATD-sf, VAE-gf corresponds to scATD-gf, and Dist-VAE corresponds to scATD-sf-dist, representing independent models developed under the scATD framework.**

Your contributions, feedback, and suggestions are highly appreciated. If you encounter any issues or have questions, feel free to reach out to the author2 (Zeyu Luo, Email: [1024226968@qq.com](mailto:1024226968@qq.com)) via the provided email addresses. Thank you for your interest in our work!

## Dataset Available

All Data can be download in figshare (https://figshare.com/articles/software/scATD/27908847).
Additionally, the model checkpoints for VAE_sf, VAE_gf, and Dist_VAE, which were pretrained on Panglao data, are also available for download on figshare (https://figshare.com/articles/software/scATD/27908847).
Finally, the transfer learning and drug resistance models can be trained by following the transfer learning instructions and code provided, and its backbone is the above pretrained VAE model.
## scATD Environment Setup

For the environment configuration for feature extraction using scFoundation or geneformer, please refer to the configuration in the 'Feature Extraction from LLM' section.

For the environment configuration of other scATD modules

1. Clone this repository or download the project files.
2. Navigate to the project directory.
3. Create a new Conda environment with Python version `>= 3.10`, then activate the environment:

```bash
conda create -n scATD-env python=3.10
conda activate scATD-env
```

  4.For CPU-only setup (if you don't need GPU acceleration):

```bash
pip install .
```

  5.(Optional) To enable GPU acceleration with CUDA (e.g., CUDA 12.1), please first install the necessary dependencies via Conda:

```bash
conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
After successfully installing the dependencies, install the current package with:

```bash
pip install .
```

For more detailed installation instructions and troubleshooting, please refer to the PyTorch website, which provides guides for setting up CUDA on different systems.

🔴 Note: To ensure the environment runs properly, you need to use an Anaconda or Miniconda environment. Both Windows and Linux support the operation of scATD. Additionally, we recommend dividing the workflow into three separate Conda environments to be used for: (1) feature extraction with scFoundation, (2) feature extraction with Geneformer, and (3) running other modules of scATD.


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
   -  Place the scFoundation model-derived feature data (`.npy` files) generated from previous steps into the `--open_path`, detail for this data please follow below instruction.

2. **Run 10-fold cross-validation for model pretraining**
Execute the following script to train the model and output the model checkpoint file (default is using the last epoch fold 1 checkponit as final model for downstream task, we have also provided pre-trained model checkpoints on Figshare):

```bash
python ./Res_VAE_pretraining/pretraining_after_hyperparameter/code/VAE_sf_Res-VAEpretraining.py \
    --open_path ./data/ \                     # Path to input features (.npy)
    --save_path_outer ./Res_VAE_retraining_after_hyperparameter/output \  # Output directory
    --open_path_conference_data ./reference_data \  # best hyperparameter file path, user should put best hyperparameter file here (Step1 output), or refer to the example data.
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
    --open_path_conference_data ./reference_data \  # best hyperparameter file path, user should put best hyperparameter file here (Step1 output), or refer to the example data.
    --file_prefix scRNA-seq_panglao \  # File naming prefix
    --epoch_start_for_loss_plot_only 1 \      # Start epoch for loss visualization
    --batch_size 128 \                        # Training batch size
    --REC_beta 10000 \                         # Reconstruction loss weight (β)
    --best_parameter_name VAE_gf_best_hyperparameters.xlsx        # hyperparameter file name
```
Besides, default is using the last epoch fold 1 checkponit as final model for downstream task, we have also provided pre-trained model checkpoints on Figshare

## Distillation VAE Pretraining (pretraining in Panglao data)

Step1 Dist-VAE hyperparameter optimization

The Optuna framework is similar to the VAE-sf hyperparameter optimization Step 1 code, we recommend that users directly use the optimal hyperparameter configuration file we provide.

Step2 VAE_sf Pretraining after hyperparameter optimization

1. **Prepare Input Data**
   -  Place the VAE-sf model-derived latent Z representation data (`.npy` files) and the Panglao raw scRNA-seq data (`.h5ad` files) into the `--open_path`, detail for this two data please follow below instruction.

2. **Run 10-fold cross-validation for model pretraining**
Execute the following script to train the model and output the model checkpoint file (default is using the last epoch fold 1 checkponit as final model for downstream task, we have also provided pre-trained model checkpoints on Figshare):

```bash
python ./Dist_VAE/Dist_VAE_pretraining_code_and_data/pretraining_after_hyperparamater/code/Dist_VAE_pretraining.py \
    --open_path ./data/ \                     # Path to input features (.npy), you should prepare VAE_sf_panglao_VAE_Embedding.npy and Panglao_raw_scRNA_seq.h5ad, please see figshare for detail.
    --save_path_outer ./Res_VAE_retraining_after_hyperparameter/output \  # Output directory
    --open_path_conference_data ./reference_data \  # best hyperparameter file path, user should put best hyperparameter file here.
    --file_prefix scRNA-seq_panglao \  # File naming prefix
    --epoch_start_for_loss_plot_only 1 \      # Start epoch for loss visualization
    --batch_size 128 \                        # Training batch size
    --REC_beta 500 \                         # Reconstruction loss weight (β)
    --best_parameter_name Dist_VAE_best_hyperparameters.xlsx \        # hyperparameter file name
    --VAE_sf_z_embedding_filename VAE_sf_panglao_VAE_Embedding.npy \  # VAE sf z embedding filename
    --scfoundation_panglao_feature_filename Panglao_raw_scRNA_seq.h5ad  # scRNA-seq feature filename
```


## Transfer Learning and Model Inference 

### VAE_sf  (BI-adain)

####   transfer learning training

First, refer to the `VAE_sf/training` directory. We use the `scfoundation_panglao_singlecell_cell_embedding.npy` dataset as the fixed target domain for domain-adaptive transfer learning during the bulk-to-Panglao stage. In the subsequent `Model Evaluation and Inference` phase, the direction is reversed—transferring from single-cell data to bulk data. The core of this process is a two-stage Bi-AdaIN transfer strategy. we placing `scfoundation_panglao_singlecell_cell_embedding.npy` inside the `./VAE_sf/training/target_transfer_data` directory. 

As the source domain, you should place the bulk dataset containing 1,280 patients—`AllbulkDEll_01B-resolution_bulk_cell_embedding_t4_resolution.npy`—and the corresponding label file—`ALL_label_binary_wf.csv`—into the `./Dist_VAE/training/data/in/` directory.

Other path configurations should follow the parameter descriptions provided below. Once the paths and dependence model file (see below) are correctly set, you can execute the code to perform Dist_VAE model domain adpative learning.

🔴 **Note:** All training and inference data (including bulk data and GSE single-cell data feature and label) used in VAE_sf are preprocessed according to the instructions provided in the *Data Preprocessing and Feature Extraction from LLM* section. The experimental 16 GSE datasets are available on Figshare.  

🔴 **Note:** For the other datasets reported in the main text, the drugs used for model training were strictly selected based on Table 1 (in main text) to ensure consistency between the dataset and the experimental specifications. Therefore, it is essential to correctly configure the `--label_mapping` parameter and to ensure that all other relevant parameters are set appropriately.


code 

```bash
python ./VAE_sf/training/code/VAE_sf_training.py --open_path ./VAE_sf/training/data/in/ --save_path ./VAE_sf/training/output/BI_Adain/DOCETAXEL --file_prefix DOCETAXEL_VAE_sf_training --epoch_start_for_loss_plot_only 1 --batch_size 64 --device_choose cuda:2 --model_configuration_path ./VAE_sf/VAE_sf_pretraining_model --learning_rate 2e-5 --weight_decay 3e-3 --num_epochs 150 --label_mapping "{\"sensitive\": 0, \"resistant\": 1}" --class_num 2 --drug_label_choose DOCETAXEL --model_parameters_file_pretraining VAE_sf_checkpoint_fold1_epoch_24.pth --VAE_augmentation_used True --SMOTE_used False --multiplier_choose auto --open_path_conference_data ./VAE_sf/training/target_transfer_data --style_alignment_file scfoundation_panglao_singlecell_cell_embedding.npy 
```

`--open_path`: The input path containing the source domain bulk training embeddings and corresponding label file.

`--save_path`: The output directory where all training results (model checkpoint, validation metrics, logs) will be saved.

`--file_prefix`: A prefix for output files, used to distinguish results from different training runs or datasets.

`--epoch_start_for_loss_plot_only`: Specifies from which epoch to start plotting the loss curve (used to avoid noisy early-stage training).

`--batch_size`: Batch size used during training.

`--device_choose`: Specifies the device to run training on (e.g., `cuda:0`, `cuda:1`, or `"cpu"`).

`--model_configuration_path`: **VAE_sf pretraining model path**, located at `./VAE_sf/VAE_sf_pretraining_model`. Please refer to the `model.txt` in this directory for instructions to download the pretrained checkpoint (e.g., from Figshare). Results of *VAE_sf hyperparameter optimization and Pretraining* section.

`--learning_rate`: The learning rate for model training.

`--weight_decay`: Weight decay (L2 regularization) to prevent overfitting.

`--num_epochs`: Number of total training epochs.

`--label_mapping`: A JSON-style dictionary that maps drug response categories (e.g., `"sensitive"` or `"resistant"`) to numeric labels for classification.

`--class_num`: The number of classification categories (e.g., 2 for binary classification).

`--drug_label_choose`: The drug name (e.g., `DOCETAXEL`) used to filter relevant samples label for this training run. Please refer to the column names in `ALL_label_binary_wf.csv` to obtain the correct spelling of drug names.

`--model_parameters_file_pretraining`: File name of the pretrained model checkpoint used for initializing the network weights. Comes from *VAE_sf Pretraining*.

`--VAE_augmentation_used`: Boolean flag indicating whether VAE-based latent augmentation is enabled during training.

`--SMOTE_used`: Boolean flag indicating whether SMOTE oversampling is applied for class imbalance. Set to `False` by default.

`--multiplier_choose`: Strategy for setting the ratio (multiplier) of the VAE augmentation. `"auto"` lets the system adaptively adjust it during training.

`--open_path_conference_data`: The input path for the target domain (Panglao scfoundation embedding embedding data) used in style alignment during transfer learning.

`--style_alignment_file`: The name of Panglao scfoundation embedding `.npy` file used as the target domain reference for style alignment during transfer learning. 



#### Model evaluation and Inference

First, refer to the `VAE_sf/inference` directory. we set `GSE140440` dataset as an example, you should place your evaluation dataset and labels (or the inference dataset) inside the `./VAE_sf/inference/data/in/` directory.

Other path configurations should follow the parameter descriptions provided below. Once the paths and dependence model file (see below) are correctly set, you can execute the code to perform model evaluation or inference.

🔴 **Note:** All training and inference data (including bulk data and GSE single-cell data feature and label) used in VAE_sf are preprocessed according to the instructions provided in the *Data Preprocessing and Feature Extraction from LLM* section. The experimental 16 GSE datasets are available on Figshare.

🔴 **Note:** For the other datasets reported in the main text, the drugs used for model training were strictly selected based on Table 1 (in main text) to ensure consistency between the dataset and the experimental specifications. Therefore, it is essential to correctly configure the `--label_mapping` parameter and to ensure that all other relevant parameters are set appropriately.


code

```bash
python ./VAE_sf/inference/code/VAE_sf_inference.py --open_path ./VAE_sf/inference/data/in/ --save_path ./VAE_sf/inference/output/BI_Adain/GSE140440 --file_prefix DOCETAXEL_GSE140440_vae_sf_infer --batch_size 128 --device_choose cuda:2 --model_configuration_path ./VAE_sf/VAE_sf_pretraining_model --label_mapping "{\"sensitive\": 0, \"resistant\": 1}" --class_num 2 --drug_label_choose label --open_path_conference_data ./VAE_sf/inference/data/VAE_sf_DAL_model_path/DOCETAXEL --model_parameters_file checkpoint_fold<your_best_fold>_final_epoch_150.pth --style_alignment_file ./VAE_sf/training/data/in/AllbulkDEll_01B-resolution_bulk_cell_embedding_t4_resolution.npy --inference_only False
```


- `--open_path`: Path to the single-cell GSE evaluation dataset. It should include both the feature `.npy` file and the corresponding label `.csv`/`.xlsx` file (unless in `inference_only` mode).
- `--save_path`: Directory where the inference results will be saved, including:
  - evaluation metrics (`*.xlsx`)
  - prediction outputs (`*_inference_label_prob_results.xlsx`)
  - AUC and PR curve figures (`*.pdf`)
- `--file_prefix`: A prefix used for naming the output files in the save path. It distinguishes runs by dataset, drug, or method.
- `--batch_size`: Batch size used during inference. Adjust according to GPU memory capacity.
- `--device_choose`: The computing device for model inference. `"cuda:<index>"` to select a specific GPU, or `"cpu"` for CPU inference.
- `--model_configuration_path`: Path to the pretraining model structure definition (e.g., `config.py`). This ensures compatibility when loading checkpoints.
- `--label_mapping`: JSON-style string that maps drug response categories (e.g., `"sensitive"` and `"resistant"`) to numeric labels. Used to align labels during evaluation.
- `--class_num`: Number of output classes for classification. For binary classification (e.g., sensitive vs. resistant), use `2`.
- `--drug_label_choose`: Column name in the label file used as ground truth (must be indicate unless in `inference_only` mode). 
- `--open_path_conference_data`: Path to the domain-adapted model checkpoint folder (result from transfer learning). contain the best-performing model for the selected drug (e.g., `DOCETAXEL`).
- `--model_parameters_file`: The filename of the best checkpoint to be used for inference. You should select the best fold (e.g., highest AUC) from the training output path (`./VAE_sf/training/output/BI_Adain/DOCETAXEL/`) and place it under the path specified by `--open_path_conference_data`.
- `--style_alignment_file`: Path to the bulk embedding `.npy` file used for feature distribution alignment during inference via the AdaIN mechanism (sc-bulk). Typically this is the same source bulk embedding used in training.
- `--inference_only:` if only inference and not conduct evaluation. set True only conduct model inference (model deploy mode),set False conduct both model inference and evaluation (model evaluation mode). Note, when set False, `--label_mapping` must be specified, and a dataset with true labels (such as the example dataset) must be provided.


🔴 Additional Note: For inference results (model predictions), using a fixed false discovery rate (FDR) threshold (e.g., 0.05) will affect the binary classification threshold. This differs from using a default fixed probability cutoff (as implemented in our experiments, controlled by the --PP_threshold parameter).

In real-world applications, especially across different models (i.e., various drug–dataset combinations), it is often necessary to simultaneously fix or constrain both the true positive rate (TPR) and Precision within a specific and acceptable range (see our precision–recall [PR] curves for empirically observed trade-off boundaries in each setting). This is because, as shown by the PR curve, optimizing a single metric (e.g., TPR or precision) in isolation can lead to significant degradation of its complementary metric (e.g., TNR or recall), often to an unacceptable extent. Such trade-offs lead to different probability thresholds and, consequently, to altered binary classification outcomes. Notably, these adjustments do not affect AUROC or AUPRC, as both are ranking-based metrics that are independent of any specific classification threshold.

It is important to note that precision–recall curves must be computed using labeled data. Therefore, in practical scenarios, we recommend referring to the models we reported for the 16 drug–disease combinations in this study as a baseline aand draw the precision–recall (PR) curve to guide the selection of a suitable decision threshold. Alternatively, one can train models on publicly available labeled datasets and use their corresponding PR curves to select a reasonable prediction probability threshold. This calibrated threshold can then be applied when performing inference on new patients requiring clinical prediction under the same drug–disease context. (Patients whose pathological characteristics are similar to those represented in the training cohort are preferred for inference)

These recommendations are intended to facilitate the clinical translation of the scATD model. However, if the goal is to reproduce the results reported in our paper—such as single-cell drug sensitivity prediction performance in terms of AUROC, AUPRC, and F1-score—or to evaluate scATD on a new labeled dataset for benchmarking purposes, then the aforementioned threshold calibration is not required. In such cases, we adopt a fixed threshold for each model type across all 16 datasets, as implemented in our code. For consistency and fair comparison, we also recommend using this default threshold when applying scATD in benchmarking experiments.


### VAE_gf (BI-adain)

####  transfer learning training

Similar to the `VAE_sf` transfer learning training section, this process follows a comparable path and execution logic, with necessary adjustments to account for differences in both the data and model architecture. Please refer to the files under the specified parameter directory for a detailed information.

code 

```bash
python ./VAE_gf/training/code/VAE_gf_training.py --open_path ./VAE_gf/training/data/in/ --save_path ./VAE_gf/training/output/BI_Adain/DOCETAXEL --file_prefix DOCETAXEL_VAE_gf_training --epoch_start_for_loss_plot_only 1 --batch_size 64 --device_choose cuda:2 --model_configuration_path ./VAE_gf/VAE_gf_pretraining_model --learning_rate 2e-5 --weight_decay 3e-3 --num_epochs 150 --label_mapping "{\"sensitive\": 0, \"resistant\": 1}" --class_num 2 --drug_label_choose DOCETAXEL --model_parameters_file_pretraining VAE_gf_checkpoint_fold1_epoch_39.pth --VAE_augmentation_used True --SMOTE_used False --multiplier_choose auto --open_path_conference_data ./VAE_gf/training/target_transfer_data --style_alignment_file geneformer_12L_panglao_singlecell_cell_embedding.npy
```
The parameter settings and their corresponding meanings are similar to those in the `VAE_sf` transfer learning training section.

#### Model evaluation and Inference
Similar to the `VAE_sf` Model evaluation and Inference section, this process follows a comparable path and execution logic, with necessary adjustments to account for differences in both the data and model architecture. Please refer to the files under the specified parameter directory for a detailed information.


code

```bash
python ./VAE_gf/inference/code/VAE_gf_inference.py --open_path ./VAE_gf/inference/data/in/ --save_path ./VAE_gf/inference/output/BI_Adain/GSE140440 --file_prefix DOCETAXEL_GSE140440_vae_gf_infer --batch_size 128 --device_choose cuda:2 --model_configuration_path ./VAE_gf/VAE_gf_pretraining_model --label_mapping "{\"sensitive\": 0, \"resistant\": 1}" --class_num 2 --drug_label_choose label --open_path_conference_data ./VAE_gf/inference/data/VAE_gf_DAL_model_path/DOCETAXEL --model_parameters_file checkpoint_fold<your_best_fold>_final_epoch_150.pth --style_alignment_file ./VAE_gf/training/data/in/ALL_expression_n_counts.npy --inference_only False
```
The parameter settings and their corresponding meanings are similar to those in the `VAE_sf` Model evaluation and Inference section.


### Dist_VAE (MMD)

#### Domain adpative learning detail for Dist_VAE


First, refer to the `Dist_VAE/training` directory. We use the `GSE140440` dataset as an example of a target domain for domain-adaptive transfer learning, placing it inside the `./Dist_VAE/inference/data/in/` directory. As the source domain, you should place the bulk dataset containing 1,280 patients—`preprocessed_all_ALL_Bulk_Dell.h5ad`—and the corresponding label file—`preprocessed_all_ALL_Bulk_Dell_info.xlsx`—into the `./Dist_VAE/training/data/in/` directory.

Other path configurations should follow the parameter descriptions provided below. Once the paths and dependence model file (see below) are correctly set, you can execute the code to perform Dist_VAE model domain adpative learning.

🔴 **Note:** All training and inference data (including bulk data and GSE single-cell data gene-expression and label) used in Dist_VAE are preprocessed according to the instructions provided in the *Data Preprocessing and Feature Extraction from LLM* section, based on the outcome of `scfoundation-step2`.

🔴 **Note:** For the other datasets reported in the main text, the drugs used for model training were strictly selected based on Table 1 (in main text) to ensure consistency between the dataset and the experimental specifications. Therefore, it is essential to correctly configure the `--label_mapping` parameter and to ensure that all other relevant parameters are set appropriately.


#### Domain adpative learning code

```bash
python ./Dist_VAE/training/code/Dist_VAE_training.py --open_path ./Dist_VAE/training/data/in/ --save_path ./Dist_VAE/training/output/MMD_Dist_VAE/GSE140440 --file_prefix GSE140440_dist_vae_training --epoch_start_for_loss_plot_only 1 --batch_size 128 --device_choose cuda:2 --model_configuration_path ./Dist_VAE/distillation_VAE_pretraining_model --model_parameters_file_pretraining checkpoint_fold1_epoch_30.pth --learning_rate 2e-5 --weight_decay 3e-3 --num_epochs 150 --label_mapping "{\"sensitive\": 0, \"resistant\": 1}" --class_num 2 --drug_label_choose DOCETAXEL  --VAE_augmentation_used True --multiplier_choose auto --open_path_conference_data ./Dist_VAE/inference/data/in/ --domainscdata_MMD preprocessed_GSE140440.h5ad --mmd_weight 0.1 --post_training true --post_training_epoch_num 20
```

- `--open_path`: The input path containing the source domain bulk training data and corresponding label file.
- `--save_path`: The output directory where all training results (model checkpoint, bulk  validation data evaluation metrics) will be saved.
- `--file_prefix`: A prefix for output files, used to distinguish results from different training runs or datasets.
- `--epoch_start_for_loss_plot_only`: Specifies from which epoch to start plotting the loss curve (useful to skip unstable early epochs).
- `--batch_size`: The batch size used during training.
- `--device_choose`: Specifies the computation device to run the training, e.g., `"cuda:0"` for the first GPU, `"cuda:2"` for the third GPU, or `"cpu"` for CPU.
- `--model_configuration_path`:  **Distillation VAE pretraining model path**, located at `./Dist_VAE/distillation_VAE_pretraining_model`. Please refer to the model.txt file in this directory for instructions download the model checkpoint from figshare. Results of *Distillation VAE Pretraining (pretraining in Panglao data)* section
- `--model_parameters_file_pretraining`: The filename of the pretrained Dist-VAE model checkpoint to initialize the model weights. Results of *Distillation VAE Pretraining (pretraining in Panglao data)* section
- `--learning_rate`: The learning rate for model training.
- `--weight_decay`: The weight decay (L2 regularization) value used to prevent overfitting.
- `--num_epochs`: Total number of epochs for training.
- `--label_mapping`: A JSON-style dictionary that maps class names (e.g., "sensitive", "resistant") to numeric labels required for training. 
- `--class_num`: Number of output classes (e.g., 2 for binary classification).
- `--drug_label_choose`: The drug name used to filter samples from the dataset for training (e.g., DOCETAXEL).  Please refer to the column names in `preprocessed_all_ALL_Bulk_Dell_info.xlsx` to obtain the correct spelling of drug names.
- `--VAE_augmentation_used`: Boolean flag indicating whether to use VAE-based data augmentation during training.
- `--multiplier_choose`: Strategy for setting the ratio (multiplier) of the VAE augmentation, auto is default.
- `--open_path_conference_data`: Path to the target domain (e.g., GSE140440 single-cell data) used in domain adaptation for computing the MMD loss.
- `--domainscdata_MMD`: Filename of the target domain (e.g., GSE140440 single-cell data)  used for MMD loss calculation.
- `--mmd_weight`: The weight assigned to the MMD loss in the total training objective. A higher value increases the influence of domain adaptation.
- `--post_training`: Boolean flag indicating whether to unfreeze and fine-tune the main Res VAE model during the final phase of training. The main Res VAE model remains frozen during the initial training epochs and is only unfrozen for the last few epochs, as specified by `--post_training_epoch_num`.
- `--post_training_epoch_num`: Number of epochs for the post-training phase if `--post_training` is set to `true`.

🔴 **Note:**  Five-fold cross-validation is performed on the bulk (cell line) level to evaluate model performance. Among the five folds, the model with the best performance can be selected for subsequent drug sensitivity prediction on the single-cell GSE dataset (the next inference step). This results are saved in the `--save_path` directory.



#### Model evaluation and Inference detail for Dist_VAE

First, refer to the `Dist_VAE/inference` directory. we set `GSE140440` dataset as an example, you should place your evaluation dataset and labels (or the inference dataset) inside the `./Dist_VAE/inference/data/in/` directory.

Other path configurations should follow the parameter descriptions provided below. Once the paths and dependence model file (see below) are correctly set, you can execute the code to perform model evaluation or inference.

🔴 **Note:** All training and inference data (including bulk data and GSE single-cell data gene-expression and label) used in Dist_VAE are preprocessed according to the instructions provided in the *Data Preprocessing and Feature Extraction from LLM* section, based on the outcome of `scfoundation-step2`.

🔴 **Note:** For the other datasets reported in the main text, the drugs used for model training were strictly selected based on Table 1 (in main text) to ensure consistency between the dataset and the experimental specifications. Therefore, it is essential to correctly configure the `--label_mapping` parameter and to ensure that all other relevant parameters are set appropriately.


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

`--model_configuration_path:` **Distillation VAE pretraining model path**, located at `./Dist_VAE/distillation_VAE_pretraining_model`. Please refer to the model.txt file in this directory for instructions download the model checkpoint from figshare. Results of *Distillation VAE Pretraining (pretraining in Panglao data)* section.

`--Dist_VAE_DAL_model_path:` Dist_VAE inference model (after domain adaptive learning ) path. Results of *Domain adpative learning code* 

`--model_inference_parameters_file` **Dist_VAE inference model checkpoint**. Please refer to the model.txt file in this directory for instructions download the model checkpoint from figshare. Results of *Domain adpative learning code* 

`--inference_only:` if only inference and not conduct evaluation. set True only conduct model inference (model deploy mode),set False conduct both model inference and evaluation (model evaluation mode). Note, when set False, `--label_mapping` must be specified, and a dataset with true labels (such as the example dataset) must be provided.



## Key Feature or Gene Identification 

```
Please refer to the Key Feature or Gene Identification module - feature attribution calculate instruction.md for detail.
```

## Comparison Experiments with Other Models

```
IF YOU ARE NOT THE REVIEWER:
We are currently undergoing publish online, so the experiment setting related to this part has not been made available. Please contact the author if needed.
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
