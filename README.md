# MLOps UvA Bachelor AI Course: Medical Image Classification Skeleton Code

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Build Status](https://github.com/yourusername/mlops_course/actions/workflows/ci.yml/badge.svg)
![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

A repo exemplifying **MLOps best practices**: modularity, reproducibility, automation, and experiment tracking.
---

## 🚀 Quick Start

### 1. Installation
Clone the repository and set up your isolated environment.

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. If you are using the snellius server, first run
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# 3. Activate the source
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install the package in "Editable" mode
pip install -e .

# 4. Install pre-commit hooks
pre-commit install
```

### 2. Verify Setup
```bash
pytest tests/
```
### 3. vLLM preprocessing
To run this model you need to run some preprocessing to create the text embeddings for the input. Read the how_to_run.md in the vllm/src/ folder and place the output of the vllm preprocessing in the folder data/raw (directly in MLOps_2026) under the name text_embeddings.pkl (very important to use exactly this name)

### 4. Dataset storage
It is trivial to have the dataset at a specific place in the structure so that config.yaml and TCGA_train.job work properly. The dataset cannot be placed on scratch-shared because this does not work properly, place the dataset in the MLOps_2026 folder in a folder named data/ and inside that folder have a folder called raw/ which contains the tcga_patient_to_cancer_type.csv, the tcga_titan_embeddings.pkl file and the text_embeddings.pkl file with these exact names. Then create a folder named data/processed. 

### 5. Splitting the data
Split the raw data using the create_split.py file in scripts/ by running this code from the MLOps_2026 folder (no deeper):
```bash
python scripts/create_split.py
```
Then check if you see test_split.csv, train_split.csv and val_split.csv in data/processed. 

### 6. Run an Experiment
```bash
python experiments/train.py --config experiments/configs/train_config.yaml
```
or run an experiment using a slurm job if you are running it on the Snellius server running this from the MLOps_2026/ folder (not deeper):
```bash
sbatch slurm_jobs/TCGA_train.job
```
or run any of the other slurm jobs to get either the optimal parameters (run_hpo.job) or run the champion (run_champion.job)

### 7. Inference
You can run an inference for a single sample on our current best model using inference.py.  
Run the following command from the folder MLOps_2026 to run inference.py after running source venv/bin/activate:  
```bash
python experiments/inference.py \
  --config experiments/configs/train_config_champion.yaml \
  --checkpoint experiments/results_champion/best_checkpoint.pt 
``` 
---

### Training configuration
The training configuration can be found in experiments/configs/ and can be changed to change hyperparameters.

### Training
To train the model run the following command from the MLOps_2026 folder (not the slurm_jobs folder):  
```bash
sbatch slurm_jobs/TCGA_train.job
```

### Checkpoints
Checkpoints are saved in experiments/results (this is in .gitignore so is only there locally) and the best checkpoint can be found in experiments/result_champion/. 

### Re-searching hyperparameters
If you want to search all possible hyperparameters again, you can run this code from the MLOps_2026 folder: 
```bash
sbatch slurm_jobs/run_hpo.job
```
and then check the output in slurm_jobs/job_outputs to compare the different configurations of hyperparameters.


## 📂 Project Structure

```text
.
├── src/ml_core/          # The Source Code (Library)
│   ├── data/             # Data loaders and transformations
│   ├── models/           # PyTorch model architectures
│   ├── solver/           # Trainer class and loops
│   └── utils/            # Loggers and experiment trackers
├── experiments/          # The Laboratory
│   ├── configs/          # YAML files for hyperparameters
│   ├── results/          # Checkpoints and logs (Auto-generated)
│   └── train.py          # Entry point for training
├── slurm_jobs/           # slurm jobs that can be run on snellius
├── data/                 # The data directory
│   ├── raw/              # Folder with the raw .pkl files
│   └── processed/        # Folder with the split csv files
├── scripts/              # Helper scripts (plotting, etc)
├── tests/                # Unit tests for QA
├── vllm/                 # vllm preprocessing of text data
├── pyproject.toml        # Config for Tools (Ruff, Pytest)
└── README.md             # Readme file with setup information
```
