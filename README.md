# Serverless GNN Training Pipeline for Malware Detection

![Status](https://img.shields.io/badge/Project-Graduation%20Thesis-blue)
![Python](https://img.shields.io/badge/Python-3.10-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C)
![Compute](https://img.shields.io/badge/Modal-Serverless%20GPU-00C100)

This repository contains the high-performance training script used in my graduation project. It leverages **Modal** to train multiple Graph Neural Network (GNN) architectures in parallel on cloud GPUs to classify executable files as Malware or Benign.

> **Main Project Repository:**  
> 🎓 [**Comparative Analysis of Graph Neural Networks Architecture for Malware Detection**](https://github.com/EsraaMagdy34/Comparative-Analysis-of-Graph-Neural-Networks-Architecture-for-Malware-Detection-.git)

## ⚡ Key Capabilities

*   **Multi-Architecture Support:**  
    Automatically trains and compares:
    *   **GCN** (Graph Convolutional Networks)
    *   **GATv2** (Graph Attention Networks v2)
    *   **GraphSAGE** (Sample and Aggregate)
*   **Parallel Execution:**  
    Uses `modal.map()` to run different model configurations simultaneously on separate **L40S GPUs**.
*   **Data Integrity:**  
    Implements a strict **Hash-Based Split** (70/20/10) to ensure no data leakage between training and testing sets (i.e., different versions of the same malware family never appear in both sets).
*   **Imbalance Handling:**  
    Native support for `WeightedRandomSampler` and `Focal Loss` to handle dataset imbalance.

## 📂 How It Works

The script (`Model_v2...py`) is a self-contained pipeline that:
1.  **Loads Data:** Reads graph embeddings (`.pt` files) from a cloud volume.
2.  **Configures Models:** Iterates through a defined list of hyperparameters.
3.  **Trains:** Runs the training loop with early stopping based on Validation F1 score.
4.  **Evaluates:** Generates Confusion Matrices, ROC-AUC scores, and Accuracy plots.
5.  **Compares:** Outputs a CSV ranking all models by performance.

## 🛠️ Usage

### Prerequisites
*   Python 3.10+
*   A [Modal](https://modal.com) account
*   A Modal Volume named `malware-data` containing the dataset.

### Running the Pipeline
To start the parallel training job:

```bash
pip install modal
modal setup
modal run Model_v2_with_GAT_and_SageConv_mod_one_modal_functionpy.py
Configuration
```
You can edit the CONFIGS list inside the script to add new experiments:

code
Python
download
content_copy
expand_less
```python
CONFIGS = [
    {
        "name": "gatv2_test_run",
        "model_type": "GATv2",
        "hidden_dim": 64,
        "batch_size": 32,
        "imbalance_method": "weighted_sampler"
    },
    # Add more configurations here...
]
```
📊 Outputs

Results are saved to the /data/results folder in the cloud volume:

metrics.json: Detailed performance logs.

model.pth: The best model weights.

training.png: Loss and Accuracy curves.

confusion_matrix.png: Visual classification performance.

comparison.csv: A final report comparing all trained models.

