# Scam Message Detection for Cost-Efficient Cloud Deployment

This project fine-tunes multilingual transformer models to detect scam and phishing-style messages with high sensitivity while staying practical for low-cost deployment. The core idea is simple: missing a scam is expensive, so the modeling workflow prioritizes recall, but the final candidate also needs a small memory footprint and predictable inference latency.

The repository documents the full experimentation path in notebooks: multi-source data collection, dataset quality checks, class balancing, label-noise reduction, parameter-efficient fine-tuning, and holdout evaluation. The current leading deployment candidate is a reduced-precision MiniLM adapter model, documented here as a QLoRA-style setup, because it offers the best overall trade-off between scam recall and model size.

## Why This Project Matters

- Scam detection is a high-impact classification problem where false negatives matter more than a small number of extra alerts.
- Production constraints matter just as much as model quality, so the project measures latency, throughput, and memory footprint in addition to accuracy and F1.
- The dataset is intentionally built from multiple public sources to improve language diversity and reduce overfitting to a single scam style.

## Technical Workflow

1. **Collect diverse scam-related corpora**
   - Unified several public datasets with different schemas into a single `text` / `labels` format.
   - Mixed email, SMS, phishing, and scam-conversation style data to broaden coverage.

2. **Profile dataset quality**
   - Checked class balance, duplicates, null rows, message length, and recurring scam vocabulary.
   - Explored common scam language such as urgency, account/security references, payment prompts, and identity-related wording.

3. **Clean the data for training**
   - Removed duplicates and null values.
   - Undersampled the majority `ham` class to reduce training bias.
   - Used Cleanlab with a TF-IDF + Logistic Regression baseline to flag likely mislabeled rows.

4. **Create reproducible train / test artifacts**
   - Saved the processed splits as Hugging Face datasets for reuse across experiments.
   - Final balanced modeling dataset:
     - `17,396` total samples
     - `14,786` train
     - `2,610` unseen holdout test

5. **Fine-tune multilingual transformer baselines**
   - `microsoft/Multilingual-MiniLM-L12-H384`
   - `distilbert-base-multilingual-cased`
   - Used LoRA adapters for parameter-efficient fine-tuning on a local Mac environment.
   - For MiniLM, loaded the base model in `float16` rather than full 32-bit precision to reduce memory footprint during training and serving.
   - Tracked experiments with MLflow for reproducibility and comparison.

6. **Evaluate for both ML quality and deployment readiness**
   - Holdout classification metrics: accuracy, precision, recall, F1
   - Serving metrics: p50 / p95 / p99 latency, throughput
   - Architecture metrics: parameter count, hidden width, layer count, memory footprint

## Model Comparison

| Model | Test Accuracy | Test Precision | Test Recall | Test F1 | p50 Latency | Throughput | Memory Footprint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM + QLoRA-style reduced precision | 97.51% | 98.02% | 97.05% | 97.53% | 27.50 ms | 129.66 req/s | 226.97 MB |
| Distil mBERT + LoRA | 97.32% | 98.16% | 96.53% | 97.33% | 13.43 ms | 231.50 req/s | 519.62 MB |

## Deployment Takeaway

MiniLM is the strongest deployment candidate in the current repo.

- It achieved the best scam recall on the unseen test set: `97.05%`.
- It also has a much smaller runtime footprint than Distil mBERT: `226.97 MB` vs `519.62 MB`.
- Part of that footprint advantage comes from loading the MiniLM base model in `float16` instead of full 32-bit precision.
- Distil mBERT is faster, but MiniLM offers a better balance for a cloud service where both sensitivity and cost matter.

In other words: Distil mBERT wins on raw inference speed, while MiniLM wins on the combination of recall, footprint, and operational efficiency.

Note: the MiniLM training notebook uses reduced-precision LoRA with `float16` weights and labels that experiment as `qlora_minilm` in MLflow. Full QLoRA usually refers to LoRA on top of quantized 4-bit base weights, which is a stronger form of compression than what is currently implemented here.

## Repo Guide

- [notebooks/01_data_preparation.ipynb](./notebooks/01_data_preparation.ipynb): multi-source data collection, schema unification, balancing, and split creation
- [notebooks/02_data_cleanning.ipynb](./notebooks/02_data_cleanning.ipynb): label cleaning and CSV export
- [notebooks/02_distil_mBERT_trainning.ipynb](./notebooks/02_distil_mBERT_trainning.ipynb): Distil mBERT LoRA training
- [notebooks/03_miniLM_trainning.ipynb](./notebooks/03_miniLM_trainning.ipynb): MiniLM QLoRA-style reduced-precision training
- [notebooks/04_miniLM_evaluation.ipynb](./notebooks/04_miniLM_evaluation.ipynb): MiniLM holdout evaluation
- [notebooks/05_distilmBERT_evaluation.ipynb](./notebooks/05_distilmBERT_evaluation.ipynb): Distil mBERT holdout evaluation

## Frameworks and Stack

- Python
- Jupyter Notebook
- Pandas
- Hugging Face Datasets
- Hugging Face Transformers
- PEFT / LoRA
- Reduced-precision `float16` loading for MiniLM
- Scikit-learn
- Cleanlab
- PyTorch
- TensorFlow
- MLflow
- Apple Silicon / MPS local training environment

## Recruiter Summary

This project shows end-to-end applied ML work, not only model training. It covers data sourcing, label-quality review, feature-level EDA, parameter-efficient transformer fine-tuning, experiment tracking, and deployment-oriented evaluation. The result is a practical scam detection pipeline designed around a real product constraint: catch as many scams as possible without making cloud serving expensive.
