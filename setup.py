from setuptools import setup

setup(
    name="scATD",
    version="0.1",
    author="Zeyu Luo",
    author_email="1024226968@qq.com",
    description="scATD is a tool for single-cell transcriptome data drug response prediction",
    install_requires=[
        'scanpy==1.10.2',
        'seaborn==0.13.2',
        'transformers==4.44.1',
        'torch==2.2.2',
        'scikit-learn==1.5.1',
        'numpy==1.26.4',
        'pandas==2.2.2',
        'openpyxl'=='3.1.5',
    ],
    extras_require={
        'cuda': [
            'torchvision==0.17.2',
            'torchaudio==2.2.2',
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',  
    packages=['VAE_gf', 'VAE_sf', 'Dist_VAE', 'survival_analysis', 'Res_VAE_pretraining', 'LLM_feature_extraction', 'feature_attribution_calculation'],
)