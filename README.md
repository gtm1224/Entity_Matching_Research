# Explanation-Augmented Entity Matching Research

This repository contains code and data supporting the reproduction of [Learning from Natural Language Explanations for
Generalizable Entity Matching (Wadhwa et. al., 2024)](https://aclanthology.org/2024.emnlp-main.352.pdf)'s implementation of Explanation-Augmented Entity Matching (EM) research using [google/flan-t5-base](https://huggingface.co/google/flan-t5-base). The codebase supports standard fine-tuning, as well as **Explanation-Augmented (EA)** training, where an LLM ([mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) via vLLM) generates explanations for matching/non-matching pairs to study effects on classification model performance.

## 1. Dependencies

**Requirements:**
*   Python 3.8+
*   CUDA-enabled GPU (Required, code assumes GPU presence)

**Install Necessary Libraries:**

```bash
pip install torch \
transformers \
datasets \
pandas \
numpy \
scikit-learn \
aiohttp \
pyyaml \
tqdm
```

*Note: To run the explanation generator (`generate.py`), you will also need access to a local or remote **vLLM** endpoint.*

*Note: See (`requirements.txt`) for full list of dependencies (including sub-dependencies), and version numbers if you run into issues using the simple command above.*

## 2. Data Download & Setup

We attempt reproduction of the original authors' work using several benchmarks: **Abt-Buy**, **Walmart-Amazon**, **iTunes-Amazon**, and **WDC** (Cameras, Computers, Shoes, Watches).

1.  **Download Data:** Ensure the CSV files (`tableA.csv`, `tableB.csv`, `train.csv`, `valid.csv`, `test.csv`) are placed in their respective folders inside `data/`.
    *   Example: `data/abt-buy/train.csv`
    *   *Note: The `Warlmart-amazon` folder name contains a typo in the repository but should be kept as-is for the scripts to resolve paths correctly.*
    *   *Note: Alternatively, the datasets can be downloaded directly from HuggingFace using the respective [matchbench](https://huggingface.co/matchbench/datasets) repositories.*

2.  **Process Walmart-Amazon:** If using Walmart-Amazon, run the processing script to format columns correctly:
    ```bash
    cd data/Warlmart-amazon
    python process_table.py tableA.csv
    python process_table.py tableB.csv
    ```

## 3. Preprocessing (Explanation Generation)

To perform Explanation-Augmented training, you must first generate explanations for the training pairs using an LLM.

1.  **Start vLLM:** Ensure a vLLM server is running (default config expects `mistralai/Mistral-7B-Instruct-v0.1` at `http://localhost:8000/v1`).

2.  **Configure Paths:**
    Open the specific YAML config file in `data/ea_data/` (e.g., `config_wdc_computers.yaml`).
    **Important:** Update the absolute paths in `input_A`, `input_B`, `input_matches`, `prompt`, and `output` to match your local machine's directory structure.

3.  **Run Generator:**
    ```bash
    cd data/ea_data
    # Example for WDC Computers
    python generate.py --config config_wdc_computers.yaml
    ```
    This will produce a new CSV file (e.g., `computers_train_ea.csv`) containing the column `explanation`.

## 4. Training

The `general_train.py` script handles fine-tuning. It supports both base model training and EA training.

### Configuration
Open `general_train.py` and modify the `main` block to select your dataset and training mode.

**For Base Model (No Explanations):**
```python
train_dir, valid_dir, test_dir, tableA_dir, tableB_dir, output_dir = get_dir_for_base_model_training("wdc-computers")
```

**For Explanation-Augmented (EA) Model:**
```python
# First get base dirs
train_dir, valid_dir, test_dir, tableA_dir, tableB_dir, output_dir = get_dir_for_base_model_training("wdc-computers")

# Point train_dir to the generated EA file
train_dir = "./data/ea_data/computers_train_ea.csv"
output_dir += "_with_ea"
```

### Run Training
```bash
python general_train.py
```
This script will:
1.  Serialize the entity pairs (and explanations if EA is enabled).
2.  Fine-tune `google/flan-t5-base`.
3.  Save checkpoints to the defined `output_dir`.

### Ablation Studies
To test robustness, you can apply corruptions to the explanations inside `general_train.py` by modifying the `preprocessing_dataset_auto` call:
*   `ablation="B"`: Shortens explanations (removes stopwords/random tokens).
*   `ablation="E"`: Corrupts explanations (replaces tokens with `<unk>`).

## 5. Evaluation

Use `flan_t5_classification.py` to evaluate trained checkpoints.

1.  Open `flan_t5_classification.py`.
2.  Set `base_dir` to your trained model folder:
    ```python
    base_dir = "./flan_t5_wdc/computers_base_model_with_ea"
    ```
3.  Set dataset you want to test
    ```python
     get_dir_for_base_model_training("Warlmart-amazon")
    ```
4.  Run the evaluation:
    ```bash
    python flan_t5_classification.py
    ```

The script will automatically load the latest checkpoint, run inference on Train, Validation, and Test sets, and print metrics (Precision, Recall, F1, Accuracy).

## 6. Results

The following tables summarize the performance (F1 Score) of our reproduction compared to the original authors' reported results.

### 1. In-Domain Supervised Training (Baseline)
Comparison of Flan-T5 (Base) trained and tested on the same dataset without explanation augmentation.

| Dataset | Ours (F1) | Author Reported (F1) |
| :--- | :--- | :--- |
| **Abt-Buy** | 91.13 | 89.92 |
| **Walmart-Amazon** | 86.58 | 87.4 |
| **iTunes-Amazon** | 94.54 | 93.09 |
| **WDC-Computers** | 94.08 | 92.08 |

### 2. Transfer Learning & Generalization
This table evaluates the model's ability to transfer knowledge across different domains, schemas, and distributions. It compares the Baseline (BL) against the Explanation-Augmented (EA) approach and two ablation studies (B: Shortened, E: Corrupted).

**Format:** `Ours (Ours Full) / Author`

| Type | Training Data | Tested On | F1 (Baseline) | F1 (EA Mistral) | Ablation B | Ablation E |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Cross-Domain** | Abt-Buy | Beer | 71.79(64.08) / 68.86 | 84.85(80.12) / 89.66 | 77.78(74.53) / 88.81 | 84.85(78.71) / 87.50 |
| **Cross-Domain** | Walmart-Amazon | Beer | 90.32(85.71) / 77.77 | 89.65(85.04) / 89.65 | 89.66(85.50) / 89.30 | 92.86(86.40) / 83.33 |
| **Cross-Domain** | WDC-Computers | WDC-Cameras | 90.01(86.10) / 73.26 | 87.24(84.16) / 93.77 | 88.71(86.71) / 91.92 | 88.19(84.98) / 90.18 |
| **Cross-Schema** | iTunes-Amazon | Walmart-Amazon | 49.93(48.80) / 20.04 | 46.32(43.52) / 43.09 | 28.84(29.14) / 40.49 | 30.44(29.12) / 25.64 |
| **Cross-Schema** | Walmart-Amazon | iTunes-Amazon | 73.24(68.75) / 51.72 | 80.60(84.70) / 75.63 | 72.72(77.91) / 73.33 | 73.91(77.13) / 76.41 |
| **Cross-Distribution** | Abt-Buy | Walmart-Amazon | 36.49(37.27) / 25.77 | 39.78(41.05) / 45.09 | 35.53(36.74) / 44.09 | 31.14(32.48) / 40.75 |
| **Cross-Distribution** | Walmart-Amazon | Abt-Buy | 76.06(75.96) / 63.75 | 71.49(71.72) / 67.52 | 62.50(64.03) / 68.99 | 67.16(65.56) / 67.55 |

> **Note:** The numbers in parentheses `( )` indicate scores obtained when testing on the **full target dataset** (combining train, validation, and test splits). The number preceding the parentheses represents the score on the standard test split.

## License
MIT License. See `LICENSE` file for details.
