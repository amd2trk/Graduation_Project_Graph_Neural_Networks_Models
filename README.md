Here is the updated **Repository Description** and **README.md**. I have customized them to highlight that this is part of your graduation project and to reflect the specific "Comparative Analysis" nature of your work.

### 1. Repository Description (GitHub About Section)

**Description:**
The official training pipeline for my graduation project: "Comparative Analysis of Graph Neural Networks for Malware Detection." Implements parallel training of GCN, GATv2, and GraphSAGE models using PyTorch Geometric and Modal serverless GPUs.

**Tags:**
`pytorch-geometric` `gnn` `malware-detection` `graduation-project` `modal` `deep-learning` `research` `cybersecurity`

---

### 2. README.md file

You can copy the code below into your `README.md`.

```markdown
# Comparative Analysis of GNN Architectures for Malware Detection

![Status](https://img.shields.io/badge/Status-Graduation%20Project-blue)
![Python](https://img.shields.io/badge/Python-3.10-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-Geometric-EE4C2C)
![Compute](https://img.shields.io/badge/Compute-Modal%20Serverless-00C100)

> **Note:** This repository contains the source code and training pipeline used for my graduation project. You can view the full project and documentation here: [**EsraaMagdy34/Comparative-Analysis-of-Graph-Neural-Networks-Architecture-for-Malware-Detection-**](https://github.com/EsraaMagdy34/Comparative-Analysis-of-Graph-Neural-Networks-Architecture-for-Malware-Detection-.git)

## 📖 Project Overview

This project evaluates the effectiveness of different Graph Neural Network (GNN) architectures in detecting malware. By treating binary executables as graphs (Control Flow Graphs), we utilize deep learning to identify malicious patterns that traditional signature-based methods might miss.

The system is built to run in a **serverless environment (Modal)**, allowing for parallel training of multiple model configurations on high-end GPUs (L40S).

## 🧠 Architectures Compared

The code implements and compares three distinct GNN layers:

1.  **GCN (Graph Convolutional Network):**
    *   *Role:* The baseline model.
    *   *Mechanism:* Aggregates features from immediate neighbors using spectral graph convolution.
2.  **GATv2 (Graph Attention Network):**
    *   *Role:* The advanced attention-based model.
    *   *Mechanism:* Uses dynamic attention mechanisms to weigh the importance of specific neighbor nodes (e.g., critical API calls) more heavily than others.
3.  **GraphSAGE (Sample and Aggregate):**
    *   *Role:* The scalable model.
    *   *Mechanism:* Samples fixed-size neighborhoods to generate node embeddings, making it efficient for large graphs.

## 🛠️ Technical Implementation

### Key Features
*   **Parallel GPU Training:** Utilizes `modal.map()` to train different configurations (e.g., GCN vs GAT) simultaneously on separate isolated containers.
*   **Strict Data Isolation:** Implements a custom data splitter that segregates data by **unique file hash** (70% Train / 20% Val / 10% Test). This prevents data leakage where different augmentations of the same malware family could bleed into the test set.
*   **Imbalance Handling:**
    *   **WeightedRandomSampler:** Oversamples the minority class (Malware) during batch creation.
    *   **Focal Loss:** Optional loss function to focus learning on hard-to-classify examples.
*   **Metrics & Visualization:** Automatically generates F1-scores, Confusion Matrices, ROC-AUC, and Training Curves for every run.

### Dataset Structure
The model expects pre-processed PyTorch Geometric data objects (`.pt` files) stored in a Modal Volume:
```text
/data
├── ebds/
│   ├── AE/              # Nodes with AutoEncoder features
│   └── BERT/            # Nodes with BERT embeddings
└── ebd_map.csv          # Metadata mapping hashes to labels
```

## 🚀 Usage

This code is designed to run on [Modal](https://modal.com/).

### 1. Setup
```bash
pip install modal
modal setup
```

### 2. Configuration
Modify the `CONFIGS` list in the script to set up your experiments:
```python
CONFIGS = [
    {
        "name": "gatv2_experiment_1",
        "model_type": "GATv2",
        "hidden_dim": 32,
        "imbalance_method": "weighted_sampler",
        ...
    },
    {
        "name": "sage_experiment_1",
        "model_type": "SAGE",
        ...
    }
]
```

### 3. Execution
Run the script to spawn the parallel workers:
```bash
modal run Model_v2_with_GAT_and_SageConv_mod_one_modal_functionpy.py
```

## 📊 Results Output

After training, the system saves the following to the cloud volume for every model:
*   **`model.pth`**: The trained model weights.
*   **`metrics.json`**: Precision, Recall, F1, and Accuracy scores.
*   **`confusion_matrix.png`**: Visual performance breakdown.
*   **`training.png`**: Loss and Accuracy over epochs.
*   **`comparison.csv`**: A summary table ranking all trained models by F1 score.

## 🔗 Citation / Reference

If you use this code or concepts, please reference the main graduation project repository:

> [Comparative Analysis of Graph Neural Networks Architecture for Malware Detection](https://github.com/EsraaMagdy34/Comparative-Analysis-of-Graph-Neural-Networks-Architecture-for-Malware-Detection-.git)

