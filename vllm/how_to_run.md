# Data Preprocessing: vLLM Narrative Label Generation

This module extracts a representative sample of text messages, filters out short or action-type posts, and utilizes an offline vLLM engine combined with schema-constrained Guided Decoding (`Outlines` backend) to generate highly structured narrative classification labels.

Make sure you have read the README file before reading this guide.

---

## Prerequisites & Dependencies

### 1. Hardware Requirements
* **GPU Memory:** This script executes **Meta-Llama-3.1-8B-Instruct**. Because it uses guided decoding constraints and an unquantized base weights file, it requires a minimum of **24GB VRAM** (e.g., a single A100 or H100 slice on Snellius).
* **Storage Space:** Ensure your home or scratch directory has at least **16GB** of free disk space to store the downloaded Hugging Face model weights.

### 2. Hugging Face Access Token
Meta-Llama-3.1 models are gated. You must accept the license terms on the Hugging Face model card and provide an access token to authorize the script download.

Set your token inside your terminal session before launching:
```bash
export HF_TOKEN="your_huggingface_access_token_here"
```
📂 Expected File Layout
Your raw input parquet file must by put into "data/raw/<your_file>.pqt and must contain at least a column called "message_text".
Before starting the pipeline, verify that your raw files and prompts match the designated project structure:

```plaintext
Thesis_Project/
├── data/
│   └── raw/
│       └── your_file.pqt            <-- Input source dataset
│   └── vllm_processed/              <-- Output directory (labeled_subset.pqt)
├── vllm/
│   └── python_scripts/
│       └── LLM_prompts/
│           └── LLM_prompt_v4.md    <-- System prompt guidelines
```
The script will automatically initialize and output results to data/vllm_processed/labeled_subset.pqt.

The "LLM_prompts/" folder contains many versions of the LLM prompt. Feel free to pick any prompt or create your own prompt. 

# Execution Guide
If you are running on a local workstation or inside an interactive compute allocation on Snellius (salloc), activate your virtual environment and call the python script:

```bash
# Activate your venv
source venv/bin/activate

# Setup the authentication token
export HF_TOKEN="your_huggingface_access_token_here"

# Run the processing script
python vllm/python_scripts/run_labeling.py
```

To run the vllm code non-interactively on a Snellius node, run the jobscript found at "slurm_jobs/vllm.job"

```bash
sbatch slurm_jobs/run_vllm.job
```
