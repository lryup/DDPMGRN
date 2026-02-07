# DDPMGRN
</p>
SIGRN: Inferring Gene Regulatory Network with Soft Introspective Variational Autoencoders

# Architecture

![DDPMGRN](/images/DDPMGRN_arc.png)

# Dependencies
- python >=3.8
- torch==2.1.0
- scanpy==1.9.1
- other detailed installation packages can be found in requirements.txt
- CUDA toolkit 11.0 or later.

# Installation
If you do not have Anaconda, please download and install Conda, then follow the steps below to create a Conda environment:
  (1) Clone the SIGRN repository from GitHub:
```
git clone https://github.com/lryup/SIGRN.git
```
  (2) Create a new environment:
```
conda create -n your_env_name python=3.8.0
  ```
  (3) Activate the environment:
```
conda activate your_env_name
 ```
  (4) Install the required packages:
 ```
pip install -r requirements.txt# It is recommended to install only the missing packages
 ```

# Data Preparation
In our study, we trained our model using all data from [BEENLINE](https://bcb.cs.tufts.edu/DAZZLE/BEELINE.zip).
You can download the datasets from the provided link. 
# Runing
The training command we used is as follows:
```
cd DDPMGRN  #Navigate to the current working directory
python run.py
```
# Result
```
0:1000_STRING:hHep
<module 'regdiffusion.data' from '/home/jxlab/lry/lry_python/phd_code/EMOGI-master/scRNA/DDPMGRN_github/regdiffusion/data/__init__.py'>
//home/jxlab/soft/miniconda3/envs/lry_torch/lib/python3.8/site-packages/torch/nn/modules/transformer.py:282: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.batch_first was not True(use batch_first for better inference performance)
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
100%|██████████████████████████████████████| 1000/1000 [00:08<00:00, 111.72it/s]
{'AUROC': 0.642506259634387, 'AUPR': 0.05243550880885535, 'AUPRR': 2.137061597013709, 'EP': 876, 'EPR': 3.966917333333333}
```



# Usage

SIGRN accepts input data in CSV, TSV format, or H5AD format as provided by Scanpy (genes in rows and cells in columns for TSV and CSV). The output of the  GRN inference task includes an adjacency matrix and various evaluation metrics, such as AUC, EPR, and AUPRR.

# Baseline methods
- Beeline https://github.com/Murali-group/Beeline/tree/master
- DeepSEM https://github.com/HantaoShu/DeepSEM
- GRN-VAE/DAZZLE https://github.com/TuftsBCB/dazzle/tree/main

# References

- Thanks to the following authors for their papers and codes.

  [1] Pratapa, A., Jalihal, A.P., Law, J.N., Bharadwaj, A., Murali, T.: Benchmarking algorithms for gene regulatory network inference from single-cell transcriptomic data. Nature methods 17(2), 147–154 (2020)

  [2] Shu, H., Zhou, J., Lian, Q., Li, H., Zhao, D., Zeng, J., Ma, J.: Modeling gene regulatory networks using neural network architectures. Nature Computational Science 1(7), 491–501 (2021)

  [3] Zhu, H., Slonim, D.: Grn-vae: A simplified and stabilized sem model for gene regulatory network inference. bioRxiv pp. 2023–01 (2023)

  
