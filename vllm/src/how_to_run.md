# 🚀 vLLM Inference & Embedding Guide (Qwen3) on Snellius

This guide explains how to perform **Batch LLM classification** (generating classifications on text using an LLM) using the Llama-3.1-8B-Instruct model on the Snellius `gpu_h100` partition.

## 📂 Project Structure
All commands should be executed from the **`src/`** directory to ensure paths remain consistent, but you can change this how you want:
```text
vllm/src/
├── data/                 # Input (.pqt) and Output (.pqt)
├── python_scripts/       # Python client scripts (.py)
├── slurm_jobs/           # SLURM submission scripts (.job)
└── how_to_run.md         # This guide
```

> **Note:** Remember to adjust the parameters in the `.job` files (such as `--time`, `--mem`, and `--cpus-per-task`) to suit the size and complexity of your own dataset. And also remember to change paths to your data, right now the script works as an example! 

---

## 📥 1. Data Requirements
All text data **must** be provided in `.pqt` (Parquet Lines) format.

* **Format Requirement:** Every piece of text information you wish to process should be structured exactly like the example below.
* **Decoder Input:** Must contain a `"prompt"` key.
* **Encoder Input:** Flexible. It looks for text in this priority: `report` → `response` → `text` → `prompt`.

**Example: `data/prompts.jsonl`**
```json
{"prompt": "What is Barrett's Oesophagus?"}
{"prompt": "List the main risk factors for developing Barrett's Oesophagus."}
{"prompt": "How is Barrett's Oesophagus typically diagnosed?"}
```

---

## ⚙️ 2. Running Batch Inference (The Decoder)
This step takes your prompts and generates text responses using the **Qwen3-4B-AWQ** model.

### Controlling "Creativity" (Temperature)
You can modify the `temperature` parameter within the `.job` script or Python call to change the nature of the output:
* **Low Temperature (e.g., 0.1 - 0.6):** Makes the model **deterministic**. Useful for factual, consistent answers. *Beware: very low values can lead to repetitive, looping text.*
* **High Temperature (e.g., 0.8 - 1.8):** Makes the model **stochastic** (creative/random). Useful for brainstorming or varied phrasing.

### Submission
```bash
sbatch slurm_jobs/batch_inference_decoder.job
```

### 📤 Decoder Output
The script produces a new `.jsonl` file in `data/`. It preserves your original prompt and adds a `"response"` field:
```json
{"prompt": "What is Barrett's Oesophagus?", "report": "Barrett's oesophagus is a condition where...", "mode": "no_think"}
```

---

## 🧠 3. Running Batch Embedding (The Encoder)
This step converts text into high-dimensional vectors using the **Qwen3-Embedding-0.6B** model.

### Task Descriptions & Instructions
The embedding script uses a `--task_description` string. This is **prepended** to every piece of text in your `.jsonl` before it is embedded. This helps the model understand the context of the vector it is creating.

**Example execution logic within the job script:**
```bash
apptainer exec --nv -B "$PWD" "$CONTAINER" \
  python3 ./python_scripts/batch_inference_encoder.py \
  --input "./data/results.jsonl" \
  --output "./data/embeddings.pkl" \
  --is_query \
  --task_description "Given the following, answer as if you were an experienced expert pathologist." \
  --concurrency 32
```

### Submission
To submit the embedding job to the cluster, run:
```bash
sbatch slurm_jobs/batch_inference_encoder.job
```

### 🔗 "Piping" Outputs
A key feature of this setup is that you can **pipe the output of the Decoder directly into the Encoder**. Since the Decoder output contains a `"report"` field and the Encoder prioritizes that field, you should aim to chain the two slurm jobs into one.

Make sure to create your own job script for this piping!

---

## 📤 4. Output Formats
| Component | Format | Content | Usage |
| :--- | :--- | :--- | :--- |
| **Decoder** | `.jsonl` | Original prompt + Model response | Human readable text |
| **Encoder** | `.pkl` | (pid -> embedding) vector dictionary | Mathematical analysis/Similarity search |

**Alignment Note:** The pids in the `.pkl` file should match the pids in the `.jsonl` exactly.

---

## 🔗 Useful Links
* **Main Model:** [Qwen/Qwen3-4B-AWQ (HuggingFace)](https://huggingface.co/Qwen/Qwen3-4B-AWQ)
* **Embedding Model:** [Qwen/Qwen3-Embedding-0.6B (HuggingFace)](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
* **vLLM Deployment:** [Qwen Documentation for vLLM](https://qwen.readthedocs.io/en/latest/deployment/vllm.html)
* **SURF Documentation:** [LLM inference on Snellius with vLLM](https://servicedesk.surf.nl/wiki/spaces/WIKI/pages/232851290/LLM+inference+on+Snellius+with+vLLM)


