### Environment Setup Instructions

First you need to create new conda environment for Key Feature or Gene Identification module

second, you should download protloc-mex1 0.0.24 ((https://pypi.org/project/protloc-mex1/)), captum== 0.6.0, torch (version: cpu) and scanpy.

(package version may differ in your computer, we courage you to search for 'protloc-mex1 0.0.24 and captum, torch cpu version, scanpy Compatible Versions’ in Chat GPT/gemmni/Grok3 search function)

OR you can reference below instruction

```bash
#1.create conda env
conda create -n myenv python=3.9
conda activate myenv

#2.install pytorch+cpu
pip install pip install torch==1.12.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html

#3.install captum==0.6.0
pip install captum==0.6.0

#4.install package version match ProtLoc-Mex1 
pip install scanpy==1.9.3 anndata==0.8.0             \
         statsmodels==0.13.5 contourpy==1.0.7
         
#5.install openpyxl
pip install openpyxl

#6.install ProtLoc-Mex1
pip install protloc_mex1
```



### Data preprocessing

Overall, the interpretability framework consists of two components: **feature-level importance computation** based on scFoundation embeddings, and **gene-level importance computation** based on RNA-seq data. The models used are the previously trained, fine-tuned and transfer learning **VAE-sf** and **Dist-VAE**, which were adapted for downstream drug sensitivity tasks. The data input logic and dependencies for the feature attribution algorithms (e.g., Integrated Gradients or GradientShap) are nearly identical to those used in the model inference for drug sensitivity prediction (the same model for **Model evaluation and Inference** section). Therefore, for both **bulk** and **single-cell** features or RNA-seq data, the preprocessing and model input logic should follow the same procedures as described in the previous data processing and model inference steps.

Specifically, **VAE-sf** is responsible for computing feature-level importance from the scFoundation embeddings, while **Dist-VAE** computes gene-level importance from RNA-seq data.

For **TCGA data** (obtained either from Figshare or directly downloaded from TCGA), the gene identifiers are encoded using **ENSEMBL gene symbols (ENGS)**. Therefore, gene name conversion is required. After conversion, the data can be processed either through scFoundation to obtain embeddings or directly used as RNA-seq input. In both cases, preprocessing steps (please follow the Data Preprocessing and Feature Extraction from LLM section, after gene name conversion) such as gene ordering alignment and log-normalization must be applied to ensure compatibility with the expected inputs of the **VAE-sf** and **Dist-VAE** model before importance computation can proceed.

Regarding TCGA data processing, we provide the corresponding scripts and gene name mapping files in the `TCGA_data_preprocess` directory. Please refer to the source code for detailed instructions. 

Importantly, in line with the paper’s intended design, the interpretability analysis—whether for bulk, TCGA, or scRNA-seq—is performed using models that were transferred from bulk to single-cell data (the same model for **Model evaluation and Inference** section). However, applying interpretability analysis directly on a non-transferred bulk-trained model is also a viable alternative, and we recommend exploring this as optional.



### Feature attribution calculation

VAE-sf methods: GradientShap

```bash
python ./feature_aggregation_value_conduct_GradientShap_general_basline.py \
  --open_path <The file path to the RNA-seq embedding .npy data from scFoundation, exp: ./VAE_sf/inference/data/in/> \
  --save_path <the specified output save path, exp: output/> \
  --file_prefix <prefix user specified,exp: TCGA_LUAD_SHAP_GradientShap_mean_VAE_aug> \
  --path_to_add <Please see the parameter descriptions below, exp: ./scATD-master> \
  --label_mapping "{\"sensitive\": 0, \"resistant\": 1}" \
  --class_num 2 \
  --open_path_conference_data <Please see the parameter descriptions below exp: ./VAE_sf/inference/data/VAE_sf_DAL_model_path/DOCETAXEL> \
  --model_parameters_file <exp: checkpoint_fold<your_best_fold>_final_epoch_150.pth> \
  --feature_attribution_display_num 10 \
  --baseline_type <Please see the parameter descriptions below, exp: mean> \
  --random_baseline_sample <! only baseline_type setting equal to random, using this parameter to control Number of random baseline samples to choose, exp: 10> \
  --shap_summary_plot_figure_size 7 7
```

**Note:** Parameters enclosed in `<>` must be set by the user. All other parameters are fixed and required for executing the code.

`--open_path`: This should be the path to a `.npy` file containing RNA-seq embeddings obtained using scFoundation (eg. bulk, single-cell, or TCGA RNA-seq data) . Note that only one `.npy` file is allowed at a time in this path.

`--save_path`: This should be the directory to save the output files, including the feature importance file `<Prefix>_shap_feature_importance.csv` and the SHAP summary plot.

`--path_to_add`: Should be set to the **absolute path** which include the `./scATD-master/VAE_sf/VAE_sf_pretraining_model`, exp: `./scATD-master`.  For detailed instructions, please refer to the section **" VAE_sf (BI-adain)"**,  **"transfer learning training `--model_configuration_path`"**.

`--open_path_conference_data`: Should be set to the same path as the best checkpoint model of `VAE_sf`(results of `./VAE_sf/training/code/VAE_sf_training.py`), used for inference. For detailed instructions, please refer to the section **" VAE_sf (BI-adain)"**,  **"Model evaluation and Inference `--open_path_conference_data`, `--model_parameters_file`"**.

`--baseline_type` control baseline, can be set as mean, zero, random, please see main article or source code for the detail meaning of this parameter 



VAE-sf methods: IG

```bash
python ./feature_aggregation_value_conduct_IG_general_basline.py \
  --open_path <The file path to the RNA-seq embedding .npy data from scFoundation, exp: ./VAE_sf/inference/data/in/> \
  --save_path <the specified output save path, exp: output/> \
  --file_prefix <prefix user specified,exp: TCGA_LUAD_SHAP_IG_mean_VAE_aug> \
  --path_to_add <Please see the parameter descriptions below, exp: ./scATD-master> \
  --label_mapping "{\"sensitive\": 0, \"resistant\": 1}" \
  --class_num 2 \
  --open_path_conference_data <Please see the parameter descriptions below exp: ./VAE_sf/inference/data/VAE_sf_DAL_model_path/DOCETAXEL> \
  --model_parameters_file <exp: checkpoint_fold<your_best_fold>_final_epoch_150.pth> \
  --feature_attribution_display_num 10 \
  --baseline_type <Please see the parameter descriptions below, exp: mean> \
  --random_baseline_sample <! only baseline_type setting equal to random, using this parameter to control Number of random baseline samples to choose, exp: 10> \
  --shap_summary_plot_figure_size 7 7
```

**Note:** Parameters enclosed in `<>` must be set by the user. All other parameters are fixed and required for executing the code. all parameter is the same as the above `feature_aggregation_value_conduct_GradientShap_general_basline.py`



### Gene attribution calculation

Dist-VAE methods: GradientShap

```bash
python ./gene_aggregation_value_conduct_GradientShap_general_basline.py \
  --open_path <The file path to the RNA-seq data, exp: ./Dist_VAE/inference/data/in/> \
  --save_path <the specified output save path, exp: output/> \
  --file_prefix <prefix user specified,exp: TCGA_LUAD_SHAP_GradientShap_mean_Dist_VAE> \
  --path_to_add <Please see the parameter descriptions below, exp: ./scATD-master> \
  --label_mapping "{\"sensitive\": 0, \"resistant\": 1}" \
  --class_num 2 \
  --open_path_conference_data <Please see the parameter descriptions below exp: ./Dist_VAE/inference/data/Dist_VAE_DAL_model_path/GSE140440/> \
  --model_parameters_file <exp: checkpoint_fold<your_best_fold>_final_epoch_150.pth> \
  --feature_name_to_gene <gene name file, exp: ./Key Feature or Gene Identification/reference_data/scfoundation_19264_gene_index.xlsx>
  --feature_attribution_display_num 10 \
  --baseline_type <Please see the parameter descriptions below, exp: mean> \
  --random_baseline_sample <! only baseline_type setting equal to random, using this parameter to control Number of random baseline samples to choose, exp: 10> \
  --shap_summary_plot_figure_size 7 7
```

**Note:** Parameters enclosed in `<>` must be set by the user. All other parameters are fixed and required for executing the code.

`--open_path`: This should be the path containing RNA-seq data (eg. bulk, single-cell, or TCGA RNA-seq data) . Note that only one  file is allowed at a time in this path.

`--save_path`: This should be the directory to save the output files, including the feature importance file `<Prefix>_shap_feature_importance.csv` and the SHAP summary plot.

`--path_to_add`: Should be set to the **absolute path** which include the `./scATD-master/Dist_VAE/distillation_VAE_pretraining_model`, exp: `./scATD-master`.  For detailed instructions, please refer to the section **"Dist_VAE (MMD)"**,  **"Domain adpative learning detail for Dist_VAE `--model_configuration_path`"**.

`--open_path_conference_data`: Should be set to the same path as the best checkpoint model of `Dist-VAE`(results of `./Dist_VAE/training/code/Dist_VAE_training.py`), used for inference. For detailed instructions, please refer to the section **" Dist_VAE (MMD)"**,  **"Model evaluation and Inference detail for Dist_VAE `--open_path_conference_data`, `--model_parameters_file`"**.

`--baseline_type` control baseline, can be set as mean, zero, random, please see main article or source code for the detail meaning of this parameter 

Dist-VAE methods: IG

```bash
python ./gene_aggregation_value_conduct_IG_general_basline.py \
 --open_path <The file path to the RNA-seq data, exp: ./Dist_VAE/inference/data/in/> \
  --save_path <the specified output save path, exp: output/> \
  --file_prefix <prefix user specified,exp: TCGA_LUAD_SHAP_IG_mean_Dist_VAE> \
  --path_to_add <Please see the parameter descriptions below, exp: ./scATD-master> \
  --label_mapping "{\"sensitive\": 0, \"resistant\": 1}" \
  --class_num 2 \
  --open_path_conference_data <Please see the parameter descriptions below exp: ./Dist_VAE/inference/data/Dist_VAE_DAL_model_path/GSE140440/> \
  --model_parameters_file <exp: checkpoint_fold<your_best_fold>_final_epoch_150.pth> \
  --feature_name_to_gene <gene name file, exp: ./Key Feature or Gene Identification/reference_data/scfoundation_19264_gene_index.xlsx>
  --feature_attribution_display_num 10 \
  --baseline_type <Please see the parameter descriptions below, exp: mean> \
  --random_baseline_sample <! only baseline_type setting equal to random, using this parameter to control Number of random baseline samples to choose, exp: 10> \
  --shap_summary_plot_figure_size 7 7
```

**Note:** Parameters enclosed in `<>` must be set by the user. All other parameters are fixed and required for executing the code. all parameter is the same as the above `gene_aggregation_value_conduct_GradientShap_general_basline.py`

### Plot

For details in the Plot module, please refer to the code sources (`./plot/`) for interpretability and visualization of feature attribution values. 

For more details on feature attribution value visualization, please refer to the *Model Interpretation* section of our previous project, [*Feature Representation for LLMs*](https://github.com/yujuan-zhang/feature-representation-for-LLMs).

After computing the attribution values (as output in files with the suffix shap_value.csv), the following scripts are used to generate visualizations for model interpretation:

1. feature_interaction_dependence_plot.py — Generates feature interaction dependence plots (Main Text, Figure 7f).

2. feature_dependence_plot_scatter.py / feature_dependence_plot_histgram.py — Produce feature attribution plots, in scatter or histogram format respectively (Main Text, Figure 7b and 7d).

3. feature_importance_value_scatter_plot.py — Visualizes feature importance Rank in scatter plot form (Main Text, Figure 7e, left panel).

4. gene_importance_value_scatter_plot.py — Displays gene-level importance Rank in scatter plot form (Main Text, Figure 7e, right panel).

5. TCGA_gene_importance_minmax_score_SHAP_force_plot.py — Generates the SHAP force plot for local interpretation (Main Text, Figure 9c).

Scripts 1–4 have approximate implementations and detailed descriptions available in the *Model Interpretation* section of our previous project, [*Feature Representation for LLMs*](https://github.com/yujuan-zhang/feature-representation-for-LLMs). Script 5 generates a SHAP force plot, which is commonly used for local-level interpretability analysis. For conceptual understanding of this method, we recommend consulting both the main text and the reference materials (including books and tutorials) linked in the *Model Interpretation* section of our previous project.

Important Notes:
Each of the above visualizations scripts 1-5 requires:

The feature matrix/RNA-seq gene expression matrix, which is the same as the **Feature OR Gene attribution calculation** input used for computing attribution values;

The corresponding attribution values (Results of **Feature OR Gene attribution calculation**).

Please ensure proper alignment between the model, input features, and the attribution values:

For example, if the attribution values were computed using VAE-sf with bulk-derived scFoundation features based on Integrated Gradients (IG) methods, then the same bulk feature matrix and corresponding IG values (feature attribution value, with the suffix shap_value.csv) should be used as input for scripts 1–5.

Similarly, if the attribution values were computed using Dist-VAE with TCGA RNA-seq gene expression matrix based on IG, then the same TCGA expression matrix and corresponding IG values should be used as input for scripts 1–5.

The same principle is also fit to single-cell RNA-seq data. Be caution input — always ensure consistency between the feature/gene matrix and the computed attributions.
