# Cultural Knowledge LLM: Fine-tuning & Adaptive RAG

This project implements a pipeline for fine-tuning a Large Language Model (Mistral-7B) to answer cultural knowledge questions about Iran, China, the UK, and the US. It handles both **Short Answer Questions (SAQ)** and **Multiple Choice Questions (MCQ)**.

The system utilizes **LoRA (Low-Rank Adaptation)** for efficient fine-tuning and implements an **Adaptive RAG (Retrieval-Augmented Generation)** system that falls back to Wikipedia or Web Search (DuckDuckGo) based on model confidence.

## 📋 Prerequisites

### Hardware
* **GPU:** A generic NVIDIA GPU with at least 16GB VRAM (e.g., A100, V100, or T4) is required.
    * *Note:* The code is configured to use `cuda` and `torch.float16`.
* **RAM:** At least 32GB system RAM is recommended for building the FAISS index.

### Software
* Python 3.10+
* Jupyter Notebook / Lab

## 🛠 Installation

1.  **Create a Virtual Environment** (Recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    Install the required Python libraries used in the notebook:
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    pip install transformers peft datasets accelerate bitsandbytes
    pip install sentence-transformers faiss-gpu  # Use faiss-cpu if no GPU for vector search
    pip install pandas numpy duckduckgo_search
    ```

## 📂 Project Structure & Data Requirements

Ensure your directory contains the following files before running the notebook.

### 1. Input Data Files
You must place the following CSV files in the root directory:
* `train_dataset_saq.csv`: Training data for Short Answer Questions.
* `test_dataset_saq.csv`: Test data for SAQ inference.
* `train_dataset_mcq.csv`: Training data for Multiple Choice Questions.
* `test_dataset_mcq.csv`: Test data for MCQ inference.

### 2. The Notebook
* `main.ipynb`: The primary script containing the entire pipeline.

## 🚀 How to Run

1.  Launch Jupyter:
    ```bash
    jupyter notebook
    ```
2.  Open `main.ipynb`.
3.  **Run All Cells**. The notebook performs the operations in the following order:

### Pipeline Stages

1.  **Environment Setup:** Loads libraries and initializes the `mistralai/Mistral-7B-Instruct-v0.2` model.
2.  **RAG Index Construction:**
    * Downloads a subset of Wikipedia.
    * Filters articles based on cultural keywords for specific countries (CN, IR, GB, US).
    * Builds a FAISS vector index (`./wiki_rag_index`) for retrieval.
3.  **SAQ Phase:**
    * **Training:** Fine-tunes the model using LoRA on the SAQ training set. Saves the adapter to `./saq-lora-adapter-clean`.
    * **Inference:** Runs on the test set using an **Adaptive Logic**:
        * *Attempt 1:* Direct LLM answer.
        * *Check:* If confidence score < 0.7, retrieve context from Wikipedia.
        * *Check:* If Wikipedia context is deemed irrelevant by the LLM, perform a **Web Search** (DuckDuckGo).
    * **Output:** Generates `saq_prediction.tsv`.
4.  **MCQ Phase:**
    * **Training:** Fine-tunes the model (new adapter) on the MCQ training set. Saves adapter to `./mcq_lora`.
    * **Inference:** Similar adaptive RAG logic as SAQ but formatted for multiple-choice selection (A/B/C/D).
    * **Output:** Generates `mcq_prediction.tsv`.

## 🧠 Methodology Details

### Model Configuration
* **Base Model:** Mistral-7B-Instruct-v0.2
* **Precision:** Float16
* **LoRA Config:** Rank (r)=16/8, Alpha=32/16, targeting query/key/value/output projections.

### Adaptive RAG Logic
The notebook uses a tiered approach to answering questions to balance speed and accuracy:
1.  **Direct Answer:** The fine-tuned model attempts to answer based on internal knowledge.
2.  **Confidence Check:** The probability of the generated tokens is averaged. If below the threshold (0.7 for SAQ, 0.8 for MCQ), external knowledge is sought.
3.  **Vector Search (Offline):** Searches the pre-built Wikipedia FAISS index.
4.  **Relevance Validator:** A specific prompt asks the LLM: *"Does the provided context contain the answer?"*
5.  **Web Search (Online):** If the offline context is insufficient, `duckduckgo_search` is triggered to find live information.

## 📤 Outputs

After execution, the following files will be generated:

| File | Description |
| :--- | :--- |
| `saq_prediction.tsv` | Tab-separated file containing IDs and textual answers for SAQ. |
| `mcq_prediction.tsv` | Tab-separated file containing IDs and boolean flags (True/False) for columns A, B, C, D. |
| `./wiki_rag_index/` | Directory containing the FAISS index and passage metadata. |
| `./saq-lora-adapter-clean/` | Saved LoRA weights for the SAQ task. |
| `./mcq_lora/` | Saved LoRA weights and checkpoints for the MCQ task. |

## ⚠️ Troubleshooting

* **CUDA OOM (Out of Memory):** If you encounter memory errors, try reducing `per_device_train_batch_size` in the `TrainingArguments` cells (e.g., reduce from 4 to 2, or 2 to 1).
* **Missing Index:** If the RAG index takes too long to build, ensure you have a stable internet connection for the initial Wikipedia dataset download. The code caches the index locally after the first run.
* **Web Search Errors:** The DuckDuckGo API is rate-limited. If web search fails, the code is designed to fallback to the existing context or direct answer.