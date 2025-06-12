# 🔍 ANADOE: Anomaly Detection with Outlier Exposure

## 🧠 Overview

**ANADOE** is a novel anomaly detection algorithm designed to train Deep Auto-Encoders (DAEs) in the presence of *contaminated training datasets*. It leverages **outlier exposure** to identify anomalous samples within the training set and optimizes a **weighted joint loss function** that considers both benign and malicious samples.

---

## 🚀 Getting Started

### Prerequisites

Ensure Python 3.6+ is installed, along with required libraries:

```bash
pip install -r requirements.txt
```

### Running an Experiment

```bash
python main.py -m MODEL -d /path/to/dataset --dataset DATASET_NAME -e 100 --batch-size 64
```

---

## ⚙️ Command-Line Arguments

| Argument | Description | Type | Default |
|----------|-------------|------|---------|
| `-m`, `--model` | Model name (`"AE","KitNet","ALAD","DUAD","NeuTraLAD",`) | `str` | **Required** |
| `-d`, `--dataset-path` | Path to dataset | `str` | **Required** |
| `--dataset` | Dataset name (`available_datasets`) | `str` | **Required** |
| `-e`, `--n-epochs` | Number of epochs | `int` | 200 |
| `--n-runs` | Number of runs | `int` | 1 |
| `--batch-size` | Training batch size | `int` | **Required** |
| `--batch-size-test` | Test batch size | `int` | None |
| `--lr` | Learning rate | `float` | 0.0001 |
| `--weight_decay` | Weight decay | `float` | 0 |
| `--test_pct` | Train/test split ratio | `float` | 0.5 |
| `--val_ratio` | Validation split ratio | `float` | 0.2 |
| `--hold_out` | Anomalous data held out | `float` | 0.0 |
| `--rho` | Anomaly ratio in training | `float` | 0.0 |
| `--pct` | % of original data to use | `float` | 1.0 |
| `--results-path` | Save directory | `str` | None |
| `--model-path` | Model weight path | `str` | `./` |
| `--test-mode` | Only run tests | `bool` | False |
| `--seed` | Random seed | `int` | 42 |

### 🔧 Model-Specific & Robustness Parameters

| Argument | Description | Default |
|----------|-------------|---------|
| `--latent-dim` | AE latent dimension | `1` |
| `--trans-type` | Transformer type (`res`/`mul`) | `"res"` |
| `--duad_r`, `--duad_p_s`, `--duad_p_0`, `--duad_num-cluster` | DUAD settings | `10`, `35`, `30`, `20` |
| `--rob`, `--rob-sup`, `--rob-reg` | Enable robust training options | `False` |
| `--rob_method` | Robust method (`refine`, `loe`, `our`, `sup`) | `"daecd"` |
| `--drop_lastbatch` | Drop last batch if incomplete | `False` |
| `--eval-test` | Evaluate test set after training | `False` |
| `--early_stopping` | Enable early stopping | `True` |
| `--warmup` | Warm-up epochs | `0` |
| `--alpha-off-set`, `--reg_n`, `--reg_a` | Regularization settings | `0`, `0.0`, `1e-3` |
| `--type_center` | Centering method (`zero`, `mean`, `learnable`) | `"zero"` |
| `--num_clusters`, `--n_aes` | number of autoencoders | `3`, `5` |

---

## 🧪 Example Usage

```bash
python main.py \
  -m ae \
  --dataset-path ./data/iot_traffic \
  --dataset iot \
  --batch-size 64 \
  -e 100 \
  --lr 0.001 \
  --rob \
  --rob_method our \
  --early_stopping
```

---

## 📁 Output

- Model checkpoints (if `--model-path` is set)
- Result logs (metrics, plots, etc.) in `--results-path`
- Performance evaluation for both clean and contaminated scenarios

---

## 🛡️ Key Features

✅ Supports multiple datasets and models  
✅ Robustness settings to handle noisy/anomalous training data   
✅ Flexible model saving/loading  
✅ Early stopping, warm-up, regularization options

---

## 📚 Citation

If you use this framework in your research, please cite the corresponding paper (to be added).

---

## 📜 License

This project is released under the MIT License. See `LICENSE` for more information.