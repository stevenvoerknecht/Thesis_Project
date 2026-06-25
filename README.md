# Bachelor's Thesis: Multi-Label Text Classification into Strategic Narratives

![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)

An advanced multi-label text classification framework designed to analyze and categorize text sequences (e.g., large Telegram datasets) into narrative taxonomies using deep transformer.

---

# Environment Setup

### 1. Installation
Clone the repository and set up your isolated virtual environment inside your project directory.

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. If you are training or running inference on the SURF Snellius cluster, load required system modules
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0

# 3. Activate the environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install the ecosystem dependencies and local package in editable mode
pip install -e .
```

### 2. Directory Initialization
Due to storage constraints, directories managed by .gitignore must be initialized manually before executing data workflows or training jobs:

```bash
mkdir -p data/raw data/processed data/vllm_processed experiments/results
```

# Inference Guide (For Researchers)
If you are a researcher looking to apply the optimized champion narrative model to a large, unseen dataset, use this streamlined pipeline.

### 1. Input Data Requirements
Your target classification text data must be structured in Parquet format (.pqt) containing at least one string column named exactly message_text containing the text_message you wish to classify.

### 2. Running Batched Prediction
Execute the optimized inference module by pointing it to the standalone champion checkpoint folder. You can tune the batch size depending on your available GPU VRAM.

Run the following command from the project root directory:

```bash
python3 scripts/inference.py \
  --model_dir experiments/result_champion \
  --data_path data/raw/your_unseen_dataset.pqt \
  --output_path data/processed/final_tagged_predictions.pqt \
  --batch_size 64 \
  --max_length 256
```

### 3. Output Format
The inference script matches your input records using Polars and appends a binary matrix mapping to the core narrative dimensions. The final output Parquet file will append the following columns with activation values (1 for active, 0 for inactive):

```text
elite_vs_mass_conflict
in_group_vs_out_group_exclusion
institutional_knowledge_denial
societal_moral_regression
imminent_acute_crisis_panic
systemic_sovereignty_revival
```

# Model Training Pipeline
Follow these steps if you are reproducing experiments, modifying the tokenizer boundaries, or running a hyperparameter search grid.

### 1. Text Preprocessing (vLLM)
Before running training loops, raw text elements must undergo sequence generation or structuring. Read the dedicated how_to_run.md documentation located in vllm/ to execute your initial data curation passes. If you are using data that has been labeled already, check the LLM prompt at "/vllm/python_scripts/LLM_prompts/LLM_prompt_v4.md" to ensure your labels match those expected by the model. 

### 2. Generating Data Splits
Once your labeled dataset is generated, run the sequence splitting script to divide data into reproducible train, validation, and testing blocks:

```bash
python3 scripts/create_split.py
``` 

Verify that train_split.pqt, val_split.pqt, and test_split.pqt have been correctly generated inside data/processed/.

### 3. Running Single Training or Hyperparameter Optimization (HPO)
Training configurations and hyperparameters are driven by the setup files within experiments/configs/. Any change in file paths, hyperparameters or model can be made here. The current model (in the config.yaml files) used is "DTAI-KULeuven/robbert-2022-dutch-base", which is only applicable for dutch data. Use "microsoft/deberta-v3-base" for english or multilingual data, or use "BAAI/bge-m3" for very long (multilingual) messages. Other models are also possible. 

To run a standard training experiment via Slurm (on the Snellius server):

```bash
sbatch slurm_jobs/train.job
```

To run a full multi-parameter hyperparameter grid search (HPO):

```bash
sbatch slurm_jobs/hpo.job
```
If, after running the HPO, new champion hyperparameters have been found, you can put these in train_config_champion.yaml and run:
```bash
sbatch slurm_jobs/champion.job
```
Running the same code on a local GPU requires you to first run some commands into your terminal before running the code:
```bash
set -e
PROJECT_ROOT="$SLURM_SUBMIT_DIR"
export PROJECT_ROOT
export PYTHONPATH=$PROJECT_ROOT/src:$PYTHONPATH
echo "PROJECT_ROOT is: $PROJECT_ROOT"
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
source "$PROJECT_ROOT/venv/bin/activate"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

After which the training experiment can be run through:
```bash
python -u $PROJECT_ROOT/experiments/train.py --config $PROJECT_ROOT/experiments/configs/train_config.yaml 
```
And the HPO grid can be run through:
```bash
python -u $PROJECT_ROOT/experiments/train.py --config $PROJECT_ROOT/experiments/configs/hpo_config.yaml
```
### 4. Experiment Metrics Tracking
The Hugging Face Trainer logs directly to both TensorBoard and MLflow backend structures simultaneously.

To view training loss progression in real-time (TensorBoard):

```bash
tensorboard --logdir experiments/results
```
To compare metrics across your HPO grid using MLflow on Open OnDemand (Snellius):

```bash
MLFLOW_CORS_ALLOWED_ORIGINS="[https://ondemand.snellius.surf.nl](https://ondemand.snellius.surf.nl)" mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --allowed-hosts "*"
```

When running on a local GPU, this command will suffice:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### 5. Checkpoints Selection
Checkpoints generated during optimization runs are managed dynamically inside experiments/results/. To preserve disk space, only the single best evaluation checkpoint per run is kept on the cluster. The final root optimized champion parameters are explicitly exported directly into experiments/result_champion/.

# 📂 Project Structure
```plaintext
.
├── src/ml_core/            # Core architecture and package logic
│   ├── data/               # Dataset preprocessing using tokenizer
│   └── utils/              # Helper utilities
├── experiments/            # The Laboratory
│   ├── configs/            # YAML configuration files for hyperparameters
│   ├── results/            # Run checkpoints and tensor logs (Auto-generated)
│   ├── result_champion/    # Root folder containing the best evaluation model weights
│   └── train.py            # Entry point for HPO and training optimization loops
├── slurm_jobs/             # Slurm cluster submission configuration scripts
│   └── job_outputs/        # Cluster terminal log files (.out / .err)
├── data/                   # The data layer (Ignored by Git except for directory map)
│   ├── raw/                # Source text Parquet structures
│   ├── processed/          # Partitioned data files (train_split.pqt, etc.)
│   └── vllm_processed/     # Output datasets from vLLM runs
├── scripts/                # Helper scripts like inference.py and create_split.py
├── vllm/                   # LLM data augmentation and preprocessing modules
│   ├── python_scripts/     # vLLM runtime logic files
│   └── slurm_jobs/         # Slurm setup configs for extraction pipelines
├── tests/                  # Framework unit tests for QA checkpoints
├── pyproject.toml          # Code linting parameters and dependencies
└── README.md               # Pipeline execution and reproduction overview
```