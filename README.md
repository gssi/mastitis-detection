# Mastitis detection

## Abstract

Mastitis is one of the most prevalent and costly diseases in dairy farming, with serious implications for animal welfare and farm sustainability. It causes a reduction in milk quantity and quality, financial losses for farmers, and increased risk of infection among herds. As prevention and monitoring become increasingly data-driven, understanding how diseases trigger and which biological indicators are most informative is essential to support effective and efficient prevention strategies. In our work, we present a modular machine learning-based approach to detect key predictors of clinical mastitis in dairy herds. In particular, we aim to address two research questions: (i) how effective are ensemble tree-based classifiers in detecting clinical mastitis and (ii) how post-hoc explainability techniques support feature space reduction while preserving predictive performance and supporting interpretability. We train and compare four ensemble classifiers (XGBoost, LightGBM, CatBoost, Random Forest), and we assess feature relevance by combining Permutation Feature Importance and SHAP interaction values techniques. Experiments on 41,104 monthly records show consistent performance across splits, with LightGBM reaching slightly higher scores across metrics. Moreover, we derive a compact feature subset through explainability analysis, which preserves predictive performance within 2\% of that obtained with the full feature space. Our findings suggest that ensemble tree classifiers trained on multi-source and heterogeneous biological data support reliable mastitis detection, and that explainability-driven feature selection promotes interpretability and scalable applications. 



##  Overview
This project proposes a modular, machine learning–based, and biologically-informed approach to detect key indicators associated to clinical mastitis in multi-source and heterogeneous biological data. 

The workflow is structured into four main steps:
1. **Unified dataset construction** to merge productive, reproductive, clinical, and anagraphic variables into a consistent dataset.  
2. **Biology-driven data processing** that includes imputation, transformations, and feature generation strategies grounded in biological plausibility.  
3. **Training of tree-based ensemble classifiers** on balanced and structured dataset. 
4. **Post-hoc feature analysis** to extract a compact and interpretable set of predictors, useful for scalable decision support system and further research.  

The main contributions:
- **Identification of predictive patterns** using consensus feature ranking across models.
- **Explainability-driven reduction of the feature space** to promote scalability.
- **Generation of interpretable and actionable outputs** enabling effective on-farm decision-making.



## Project structure
```
mastitis-detection/
│── classifier/ # Trained ML models
│ ├── cat_model.pkl # CatBoost trained model
│ ├── lgbm_model.pkl # LightGBM trained model
│ ├── rf_model.pkl # Random Forest trained model
│ └── xgb_model.pkl # XGBoost trained model
│
│── dataset_assembly/ # Dataset construction steps
│ ├── anagraphic_step.py # Handle animal demographic data
│ ├── calving_step.py # Handle calving and reproduction records
│ ├── diseases_step.py # Process disease history and labels
│ ├── ele_conductivity_step.py # Process electrical conductivity measurements
│ ├── functional_check_step.py # Include functional check data
│ └── lactose_step.py # Process lactose concentration data
│
│── ml_process/ # Machine learning workflow
│ ├── analyzer.py # Post-hoc analysis 
│ ├── classifier_module.py # Train and evaluate ML classifiers
│ ├── dataset_assembly.py # Assemble datasets from different sources
│ ├── feature_engineer.py # Feature engineering (temporal windows-based) 
│ ├── imputer.py # Missing value imputation 
│ ├── sampler.py # Data sampling / balancing
│ └── transformer.py # Biological-based data construction
│
│── output/ # Generated results and reports
| ├── imputation_report.txt # A quantitative report of imputation results
│ ├── cat_report.txt # CatBoost performance report
│ ├── lgbm_report.txt # LightGBM performance report
│ ├── rf_report.txt # Random Forest performance report
│ ├── xgb_report.txt # XGBoost performance report
│ └── feature_summary.txt # Table of key indicators found
│
│── temporary_datasets/ # Placeholder for intermediate datasets
│ └── balanced_dataset.parquet # it's the balanced dataset ready for split -> classifcation + post-hoc analysis (see the process_run.ipynb)
│
│── libraries.py # Shared useful libraries
│── process_run.ipynb # Jupyter notebook for usage
│── requirements.txt # Python dependencies
```


## Installation

This project has been tested with **Python 3.11.9+** on Windows 11 Pro.  
Make sure you have `pip` available in your environment.
Clone the repository directly through Git. If you don’t have Git installed, you can:  
  - [Download Git](https://git-scm.com/downloads) and install it, **or**  
  - Download the repository as a ZIP file from GitHub.

```bash
# 1. Clone the repository
git clone https://github.com/gssi/mastitis-detection.git

# 2. Move into the project folder
cd mastitis-detection

# 3. (Recommended) Create a virtual environment (e.g. "my_venv")

# On Windows (PowerShell)
python -m venv my_venv

# On Linux/macOS/WSL
python3 -m venv my_venv

# NOTE for Linux/WSL users:
# If you see an error like "ensurepip is not available",
# install the venv module with:
#   sudo apt update
#   sudo apt install python3-venv -y
# (use the correct version: e.g. python3.12-venv if python3 --version is 3.12)
# Then recreate the environment with:
#   python3 -m venv my_venv

# 4. Activate the environment

# On Windows (PowerShell)
my_venv\Scripts\activate

# On Linux/macOS/WSL
source my_venv/bin/activate

# 5. Install Python dependencies
pip install -r requirements.txt   # use "pip3" if required on Linux/Mac

# (To deactivate the environment)
deactivate

```



