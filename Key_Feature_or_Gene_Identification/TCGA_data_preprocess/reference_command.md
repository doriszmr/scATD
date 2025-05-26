TCGA_LUAD preprocess

```bash
python ./code/data_preprocessed/data_preprocessed.py --open_path <see picture below, you can download from our figshare-TCGA_data> --save_path <your save path> --file_prefix <exp: TCGA_LUAD> --gene_ID Ensembl_ID --feature_name_to_gene <gene name file, exp: ./Key Feature or Gene Identification/reference_data/scfoundation_19264_gene_index.xlsx> --gene_shift true --gene_ENG_mapping_file mart_export.txt 
```

![image-20250526230806208](C:\Users\qq102\AppData\Roaming\Typora\typora-user-images\image-20250526230806208.png)

after conduct above code, you can get gene conversion h5ad data <exp: TCGA_LUAD_19264_preprocessed.h5ad> and misconduct report file. 

!! note: this processed gene conversion h5ad data is also equal to the scFoundation preprocessed h5ad data (the results of step2 of scFoundation preprocessed), detail can be refer to the source code. Hence you can using this data directly for feature embedding (run scFoundation  embedding code in step3) or using for Dist-VAE gene importance calculation.

**Note:** This processed gene-converted `.h5ad` file is equivalent to the preprocessed data generated in **Step 2** of the **scFoundation** **Data Preprocessing and Feature Extraction from LLM** pipeline. For more details, please refer to the source code. Therefore, you can use this data directly for **feature embedding** (by running the **Step 3 embedding code** in scFoundation) or for **gene importance calculation** using **Dist-VAE**.

