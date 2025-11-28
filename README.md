# Commit Message Generation — Comparative Study

This repository contains the experimental materials, code, and results used for the thesis project on automated commit message generation.  
The study evaluates and compares three representative models — **CoDiSum**, **RACE**, and **KADEL** — using two datasets and a unified evaluation framework.

---

## Models Analyzed

- **CoDiSum** (Xu et al., 2019): RNN-based model with structural + semantic input and a copy mechanism.
- **RACE** (Shi et al., 2022): Transformer model guided by retrieval of semantically similar commits.
- **KADEL** (Tao et al., 2024): Knowledge-aware training with denoising based on commit quality.

---

## Dataset

- **D1**: CoDiSum legacy dataset (reconstructed).
- **D2**: Subset of the **MCMD** dataset (Java), including edit actions and commit similarity pairs.

---

## Setup & Dependencies

This project was executed on **Google Colab Pro** with the following configurations:

### Common tools
- Python ≥ 3.8
- TensorFlow 2.20.0
- PyTorch 2.3.0
- Transformers (v4.x)
- NLTK, SacreBLEU, ROUGE, METEOR
- TensorBoard, tqdm, scikit-learn

# Common tools (Python ≥ 3.8, TensorFlow, etc.)

# For CoDiSum
pip install nltk sacrebleu rouge-score pycocoevalcap tensorflow-text

# For RACE
pip install torch==2.3.0 transformers==4.42.3 sacrebleu prettytable tensorboardX

# For KADEL
python3.8 -m pip install torch==1.9.0 transformers==4.8.2 scikit-learn==0.24.2
python3.8 -m pip install tree-sitter==0.2.2 protobuf==3.20.3 sentencepiece wandb

# Evaluation
Metrics used: BLEU, ROUGE-L, METEOR
Multiple training seeds per model
Reproducibility ensured via logs, config snapshots, and seed control
Visualizations available in /plots_metrics

# Project Structure
/CoDiSum, /RACE, /KADEL: model-specific scripts, data, and results
/plots_metrics: statistical comparisons and chart generation
stats_race_vs_kadel.csv: aggregated results
*.py: scripts for training, evaluation, preprocessing, and metrics

# Thesis
The full thesis is available in /Tesi_Magistrale.pdf and includes:
Motivation and related work
Experimental design
Model analysis and reproducibility
Evaluation and discussion