# Mastitis detection

##  Overview
This project proposes a modular, machine learning–based, and biologically-informed system for detecting clinical mastitis–related indicators in multi-source biological data.  

The system integrates four key steps:
1. **Unified dataset construction** --> merging productive, reproductive, clinical, and anagraphic variables into a consistent dataset.  
2. **Biology-guided data processing** --> applying imputation, transformations, and feature generation strategies grounded in biological plausibility.  
3. **Classifier training** --> leveraging balanced dataset to train multiple tree-based ensemble classifiers (Random Forest, LightGBM, CatBoost, XGBoost).  
4. **Post-hoc feature analysis** --> extracting a compact and interpretable set of predictors, useful for decision support and further research.  

The system design follows four guiding principles:
- **Model consensus** --> evaluating agreement across multiple classifiers to support robust predictions.  
- **Domain-aligned interpretability** --> selecting features that are biologically meaningful and comprehensible for veterinary practitioners.  
- **Actionable decision support** --> producing outputs relevant for both farm-level intervention and scientific research.  
- **Explainability-driven dimensionality reduction** --> supporting  simplified, interpretable, and scalable deployment.


## Project structure
```
mammary_diseases_indicators/
│── classifier/ # Pre-trained ML models
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
│ ├── analyzer.py # Analyze model results and metrics
│ ├── classifier_module.py # Train and evaluate ML classifiers
│ ├── dataset_assembly.py # Assemble datasets from different sources
│ ├── feature_engineer.py # Feature extraction and engineering
│ ├── imputer.py # Missing value imputation strategies
│ ├── sampler.py # Data sampling / balancing
│ └── transformer.py # Data transformations
│
│── output/ # Generated results and reports
│ ├── cat_report.txt # CatBoost performance report
│ ├── lgbm_report.txt # LightGBM performance report
│ ├── rf_report.txt # Random Forest performance report
│ ├── xgb_report.txt # XGBoost performance report
│ └── feature_summary.txt # Feature importance summary
│
│── temporary_datasets/ # Placeholder for intermediate datasets
│ └── balanced_dataset.parquet # it's the balanced dataset ready for split -> classifcation + post-hoc analysis (see the process_run.ipynb)
│
│── libraries.py # Shared utility functions
│── process_run.ipynb # End-to-end Jupyter notebook for usage
│── requirements.txt # Python dependencies
│── .gitignore
│── .git/
```


## Installation

This project has been tested with **Python 3.11.9+** on Windows 11 Pro.  
Make sure you have `pip` available in your environment.
You need Git to clone the repository directly. If you don’t have Git installed, you can:  
  - [Download Git](https://git-scm.com/downloads) and install it, **or**  
  - Download the repository as a ZIP file from GitHub.

```bash
# 1. Clone the repository
git clone https://github.com/davi94cs/mammary_diseases_indicators.git

# 2. Move into the project folder
cd mammary_diseases_indicators

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



