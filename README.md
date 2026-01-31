# Cultural Knowledge LLM: Fine-tuning & Adaptive RAG

This project implements a pipeline for fine-tuning a Large Language Model (Mistral-7B) to answer cultural knowledge questions about Iran, China, the UK, and the US. It handles both **Short Answer Questions (SAQ)** and **Multiple Choice Questions (MCQ)**.

The system utilizes **LoRA (Low-Rank Adaptation)** for efficient fine-tuning and implements an **Adaptive RAG (Retrieval-Augmented Generation)** system that falls back to Wikipedia or Web Search (DuckDuckGo) based on model confidence.

## 📋 Prerequisites

### Hardware
* **GPU:** A NVIDIA GPU with at least 16GB VRAM (e.g., A100, V100, or T4) is required.
    * *Note:* The code is configured to use `cuda` and `torch.float16`.
* **RAM:** At least 32GB system RAM is recommended for building the FAISS index.

### Software
* Python 3.10+
* Jupyter Notebook / Lab

## 🛠 Installation

1.  **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

2.  **Install Dependencies**:
    Install the required Python libraries used in the notebook:
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    pip install transformers peft datasets accelerate bitsandbytes
    pip install sentence-transformers faiss-gpu
    pip install pandas numpy ddgs
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
3.  **Run All Cells**.

## 📤 Outputs

After execution, the following files will be generated:

| File | Description |
| :--- | :--- |
| `saq_prediction.tsv` | Tab-separated file containing IDs and textual answers for SAQ. |
| `mcq_prediction.tsv` | Tab-separated file containing IDs and boolean flags (True/False) for columns A, B, C, D. |
| `./wiki_rag_index/` | Directory containing the FAISS index and passage metadata. |
| `./saq-lora-adapter-clean/` | Saved LoRA weights for the SAQ task. |
| `./mcq-lora-adapter-clean/` | Saved LoRA weights for the MCQ task. |
