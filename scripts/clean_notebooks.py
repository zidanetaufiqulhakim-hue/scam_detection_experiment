import json
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def md(text: str):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def set_source(cell: dict, text: str):
    new_cell = deepcopy(cell)
    new_cell["source"] = text.splitlines(keepends=True)
    return new_cell


def load_notebook(name: str):
    path = NOTEBOOK_DIR / name
    return path, json.loads(path.read_text())


def save_notebook(path: Path, notebook: dict):
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + "\n")


def first_line(cell: dict) -> str:
    return "".join(cell.get("source", [])).splitlines()[0] if cell.get("source") else ""


def clean_01():
    path, nb = load_notebook("01_data_preparation.ipynb")
    if nb["cells"] and first_line(nb["cells"][0]) == "# Data Preparation Pipeline":
        return
    old = nb["cells"]
    nb["cells"] = [
        md(
            "# Data Preparation Pipeline\n\n"
            "This notebook builds a unified scam detection dataset from several public sources. "
            "The goal is to increase linguistic and scenario diversity so the final classifier can "
            "generalize better to real-world scam messages instead of overfitting to a single source."
        ),
        set_source(
            old[1],
            "# Core libraries used across the data assembly workflow\n"
            "import datasets\n"
            "import pandas as pd\n",
        ),
        md(
            "## 1. Collect and Standardize Data\n\n"
            "Each source uses a different schema, so the first step is to load them, map the text and label "
            "columns into one format, and concatenate everything into a single dataframe."
        ),
        old[2],
        md(
            "## 2. Normalize the Working Schema\n\n"
            "The downstream Hugging Face training pipeline expects a simple `text` / `labels` structure. "
            "This cell keeps only the required fields and renames them into that format."
        ),
        set_source(
            old[5],
            "# Keep only the fields needed for classification and adopt a consistent schema\n"
            "final_dataset = final_dataset[['text', 'label']]\n"
            "final_dataset = final_dataset.set_axis(['text', 'labels'], axis=1)\n"
            "final_dataset.head()\n",
        ),
        md(
            "## 3. Inspect Label Balance\n\n"
            "Scam detection is a recall-sensitive task, but a heavily imbalanced training set can bias the model "
            "toward predicting the majority class. I check the distribution before applying any balancing strategy."
        ),
        old[7],
        md(
            "The merged dataset is strongly skewed toward `ham`, so I rebalance it before training. "
            "That keeps the fine-tuning stage focused on discriminating scam patterns rather than learning the prior class frequency."
        ),
        md(
            "## 4. Remove Duplicates and Nulls\n\n"
            "Before training, I clean out duplicate rows and missing records to avoid wasting capacity on repeated samples "
            "or invalid text."
        ),
        set_source(
            old[10],
            "# Remove duplicate and empty records before balancing the dataset\n"
            "final_dataset.drop_duplicates(inplace=True)\n"
            "final_dataset.dropna(inplace=True)\n"
            "final_dataset.isnull().sum()\n"
            "final_dataset.duplicated().sum()\n",
        ),
        md("The cleaned merged dataset has no remaining null values or duplicates."),
        md(
            "## 5. Undersample the Majority Class\n\n"
            "I undersample `ham` to create a balanced training set. This is a pragmatic choice for local experimentation on a Mac "
            "because it reduces training cost while keeping enough signal from both classes."
        ),
        set_source(
            old[13],
            "from sklearn.utils import resample\n\n"
            "def undersample(df, target_col):\n"
            "    # Downsample the majority class to match the minority class size\n"
            "    majority = df[df[target_col] == 'ham']\n"
            "    minority = df[df[target_col] == 'fraud']\n\n"
            "    majority_downsampled = resample(\n"
            "        majority,\n"
            "        replace=False,\n"
            "        n_samples=len(minority),\n"
            "        random_state=42,\n"
            "    )\n\n"
            "    balanced_df = pd.concat([majority_downsampled, minority])\n"
            "    return balanced_df\n\n"
            "balanced_dataset = undersample(final_dataset, 'labels')\n",
        ),
        set_source(
            old[14],
            "import seaborn as sns\n"
            "import matplotlib.pyplot as plt\n\n"
            "def plot_class_distribution(df):\n"
            "    sns.countplot(x='labels', data=df)\n"
            "    plt.title('Class Distribution After Balancing')\n"
            "    plt.xlabel('labels')\n"
            "    plt.ylabel('Count')\n"
            "    plt.show()\n"
            "    return df['labels'].value_counts()\n\n"
            "plot_class_distribution(balanced_dataset)\n",
        ),
        md(
            "## 6. Sanity-Check Semantic Separation\n\n"
            "I use sentence embeddings to estimate whether the two label groups collapse into each other semantically. "
            "This is not a formal validation step, but it helps catch obvious labeling problems early."
        ),
        old[17],
        md(
            "## 7. Explore Fraud Language Patterns\n\n"
            "Understanding the most frequent scam terms helps explain what the model is likely to learn and whether the dataset "
            "contains realistic urgency, payment, identity, and account-compromise language."
        ),
        old[19],
        md(
            "## 8. Create Train and Test Splits\n\n"
            "After balancing, I shuffle the dataset and create a holdout split. This preserves a clean unseen set for final evaluation."
        ),
        set_source(
            old[21],
            "# Shuffle the balanced dataset before creating the holdout split\n"
            "df_shuffled = balanced_dataset.sample(frac=1, random_state=42).reset_index(drop=True)\n\n"
            "# Use 85% of the data for training and 15% for the final unseen evaluation set\n"
            "split_index = int(0.85 * len(df_shuffled))\n"
            "train = df_shuffled.iloc[:split_index]\n"
            "test = df_shuffled.iloc[split_index:]\n\n"
            "print(f'Train size: {len(train)}')\n"
            "print(f'Test size: {len(test)}')\n",
        ),
        md(
            "## 9. Key Findings\n\n"
            "- Raw merged dataset: 30,480 samples\n"
            "- Original label mix: 21,274 ham / 9,206 fraud\n"
            "- Balanced modeling dataset: 17,396 samples\n"
            "- Final split: 14,786 train / 2,610 test\n\n"
            "These results show why balancing and source diversity both matter: the raw corpus is useful for coverage, "
            "but the training set needs tighter control to support stable fine-tuning."
        ),
        md(
            "## 10. Persist the Processed Datasets\n\n"
            "The final step saves the train and test sets in Hugging Face Dataset format so the training and evaluation notebooks "
            "can load them without repeating preprocessing."
        ),
        set_source(
            old[25],
            "train_ds = datasets.Dataset.from_pandas(train)\n"
            "test_ds = datasets.Dataset.from_pandas(test)\n"
            "train_ds.save_to_disk('/Users/zhakim/Documents/ML_Project/scam_detection/experiment/data/processed/train_ds')\n"
            "test_ds.save_to_disk('/Users/zhakim/Documents/ML_Project/scam_detection/experiment/data/processed/test_ds')\n",
        ),
    ]
    save_notebook(path, nb)


def clean_02():
    path, nb = load_notebook("02_data_cleanning.ipynb")
    if nb["cells"] and first_line(nb["cells"][0]) == "# Data Cleaning Workflow":
        return
    old = nb["cells"]
    nb["cells"] = [
        md(
            "# Data Cleaning Workflow\n\n"
            "This notebook turns the raw phishing email corpus into a cleaner modeling dataset. "
            "The focus is to reduce label noise and class imbalance before training."
        ),
        set_source(
            old[1],
            "# Data processing, visualization, and label-quality utilities\n\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n\n"
            "from sklearn.metrics.pairwise import cosine_similarity\n"
            "from sklearn.feature_extraction.text import TfidfVectorizer\n"
            "from sklearn.model_selection import cross_val_predict\n"
            "from sklearn.linear_model import LogisticRegression\n"
            "from sklearn.utils import resample\n"
            "from cleanlab.filter import find_label_issues\n",
        ),
        md(
            "## 1. Load the Raw Email Dataset\n\n"
            "I start from the original phishing email CSV and inspect the first rows to verify the schema before cleaning."
        ),
        old[2],
        md(
            "## 2. Rebalance the Classes\n\n"
            "The original email dataset contains more non-scam messages than scam messages. "
            "I undersample the majority class so later experiments are not dominated by `ham`."
        ),
        set_source(
            old[4],
            "# Visualize the class distribution before balancing\n"
            "def plot_class_distribution(df):\n"
            "    sns.countplot(x='is_scam', data=df)\n"
            "    plt.title('Class Distribution')\n"
            "    plt.xlabel('Is Scam')\n"
            "    plt.ylabel('Count')\n"
            "    plt.show()\n"
            "    return df['is_scam'].value_counts()\n\n"
            "plot_class_distribution(raw_df)\n",
        ),
        set_source(
            old[5],
            "def undersample(df, target_col):\n"
            "    # Match the majority class to the minority class size\n"
            "    majority = df[df[target_col] == 0]\n"
            "    minority = df[df[target_col] == 1]\n\n"
            "    majority_downsampled = resample(\n"
            "        majority,\n"
            "        replace=False,\n"
            "        n_samples=len(minority),\n"
            "        random_state=42,\n"
            "    )\n\n"
            "    balanced_df = pd.concat([majority_downsampled, minority])\n"
            "    plot_class_distribution(balanced_df)\n"
            "    return balanced_df\n\n"
            "balanced_df = undersample(raw_df, 'is_scam')\n",
        ),
        set_source(
            old[6],
            "# Shuffle after balancing so training is not affected by class ordering\n"
            "balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)\n"
            "balanced_df.head(10)\n",
        ),
        md(
            "## 3. Detect Potentially Mislabeled Samples\n\n"
            "Scam datasets often include noisy labels. I use Cleanlab on top of a simple TF-IDF + Logistic Regression baseline "
            "to flag suspicious rows before committing them to the final training corpus."
        ),
        old[9],
        old[10],
        md(
            "## 4. Remove Label Issues and Null Records\n\n"
            "After identifying suspicious rows, I drop them and perform one final null check so the saved dataset is training-ready."
        ),
        set_source(
            old[11],
            "# Remove rows flagged as likely label issues\n"
            "drop_mislabeled_df = drop_mislabeled_data(balanced_df, label_issues)\n",
        ),
        old[12],
        set_source(
            old[14],
            "# Remove any residual null rows before exporting the cleaned dataset\n"
            "drop_mislabeled_df = drop_mislabeled_df.dropna().reset_index(drop=True)\n"
            "print('Null values after dropping rows with nulls:')\n"
            "drop_mislabeled_df.isnull().sum()\n"
            "print('Value counts after dropping nulls:')\n"
            "drop_mislabeled_df.value_counts('is_scam')\n",
        ),
        md(
            "## 5. Export the Cleaned CSV\n\n"
            "This file becomes the cleaned tabular artifact for downstream training and reproducibility."
        ),
        set_source(
            old[15],
            "# Save the cleaned dataset for reuse in later experiments\n"
            "drop_mislabeled_df.to_csv('../data/processed/cleaned_phishing_email_dataset.csv', index=False)\n",
        ),
    ]
    save_notebook(path, nb)


def clean_03():
    path, nb = load_notebook("03_distil_mBERT.ipynb")
    if nb["cells"] and first_line(nb["cells"][0]) == "# Distil mBERT Fine-Tuning":
        return
    old = nb["cells"]
    nb["cells"] = [
        md(
            "# Distil mBERT Fine-Tuning\n\n"
            "This notebook benchmarks `distilbert-base-multilingual-cased` with LoRA for binary scam detection. "
            "The objective is to test whether a multilingual distilled encoder can retain high recall while remaining deployable."
        ),
        set_source(
            old[1],
            "# Training, PEFT, and experiment tracking dependencies\n\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import numpy as np\n"
            "import datasets\n\n"
            "from transformers import (\n"
            "    AutoModelForSequenceClassification,\n"
            "    AutoTokenizer,\n"
            "    TrainingArguments,\n"
            "    Trainer,\n"
            ")\n"
            "from peft import LoraConfig, get_peft_model\n"
            "from datasets import Dataset\n"
            "import mlflow\n"
            "import mlflow.transformers\n"
            "from mlflow.tracking import MlflowClient\n",
        ),
        md(
            "## 1. Load the Prepared Training Set\n\n"
            "I reuse the balanced dataset generated in the preparation notebook so each model comparison starts from the same train/test foundation."
        ),
        old[3],
        md(
            "## 2. Create a Validation Split\n\n"
            "The holdout validation set is used to monitor training quality and select the best checkpoint based on recall."
        ),
        set_source(
            old[7],
            "# Split the training data again so model selection uses a clean validation subset\n"
            "ds_split = ds.train_test_split(test_size=0.1, seed=42)\n\n"
            "ds_train = ds_split['train']\n"
            "ds_val = ds_split['test']\n\n"
            "print(f\"Train Set: {len(ds_train)}, {ds_train.to_pandas()['labels'].value_counts()}\")\n"
            "print(f\"Validation Set: {len(ds_val)}, {ds_val.to_pandas()['labels'].value_counts()}\")\n",
        ),
        md(
            "## 3. Estimate Sequence Length\n\n"
            "I inspect the 90th percentile token length to choose a truncation limit that is large enough for most messages but still cost-aware."
        ),
        old[8],
        md(
            "## 4. Encode Labels and Tokenize Text\n\n"
            "The labels are converted to integers and the text is tokenized with the Distil mBERT tokenizer using a 256-token cap."
        ),
        set_source(
            old[10],
            "# Convert string labels into integer targets expected by the trainer\n"
            "str2int = {'fraud': 1, 'ham': 0}\n\n"
            "encoded_ds_train = ds_train.map(lambda x: {'labels': str2int[x['labels']]})\n"
            "encoded_ds_val = ds_val.map(lambda x: {'labels': str2int[x['labels']]})\n\n"
            "# Tokenize with a fixed maximum length to keep training predictable on local hardware\n"
            "distilbert_model_id = 'distilbert-base-multilingual-cased'\n"
            "tokenizer = AutoTokenizer.from_pretrained(distilbert_model_id)\n\n"
            "def tokenize_function(examples):\n"
            "    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=256)\n\n"
            "tokenized_ds_train = encoded_ds_train.map(tokenize_function, batched=True)\n"
            "tokenized_ds_val = encoded_ds_val.map(tokenize_function, batched=True)\n\n"
            "print(tokenized_ds_train.features)\n"
            "print(f'Train Set:{tokenized_ds_train}\\n')\n"
            "print(tokenized_ds_val.features)\n"
            "print(f'Validation Set:{tokenized_ds_val}\\n')\n",
        ),
        md(
            "## 5. Apply LoRA and Fine-Tune\n\n"
            "Only a small set of low-rank adapters is trained instead of updating the full base model. "
            "This keeps experimentation feasible on a Mac while preserving strong classification performance."
        ),
        set_source(
            old[13],
            "from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score\n\n"
            "distilbert_model_id = 'distilbert-base-multilingual-cased'\n"
            "bert_base = AutoModelForSequenceClassification.from_pretrained(distilbert_model_id, num_labels=2)\n\n"
            "# Add LoRA adapters instead of fine-tuning every weight\n"
            "lora_config = LoraConfig(\n"
            "    r=8,\n"
            "    lora_alpha=16,\n"
            "    lora_dropout=0.1,\n"
            "    bias='none',\n"
            "    task_type='SEQ_CLS',\n"
            "    target_modules=['q_lin', 'k_lin', 'v_lin', 'out_lin'],\n"
            ")\n"
            "lora_distilmbert = get_peft_model(bert_base, lora_config)\n\n"
            "def compute_metrics(eval_pred):\n"
            "    logits, labels = eval_pred\n"
            "    predicted_class = np.argmax(logits, axis=-1)\n"
            "    return {\n"
            "        'accuracy': accuracy_score(labels, predicted_class),\n"
            "        'precision': precision_score(labels, predicted_class),\n"
            "        'recall': recall_score(labels, predicted_class),\n"
            "        'f1_score': f1_score(labels, predicted_class),\n"
            "    }\n\n"
            "training_args = TrainingArguments(\n"
            "    output_dir='./results_lora_distil-mBert',\n"
            "    num_train_epochs=4,\n"
            "    per_device_train_batch_size=16,\n"
            "    per_device_eval_batch_size=16,\n"
            "    weight_decay=0.01,\n"
            "    save_strategy='best',\n"
            "    greater_is_better=True,\n"
            "    load_best_model_at_end=True,\n"
            "    metric_for_best_model='recall',\n"
            "    fp16=False,\n"
            "    bf16=True,\n"
            ")\n\n"
            "trainer_lora_distilmBERT = Trainer(\n"
            "    model=lora_distilmbert,\n"
            "    args=training_args,\n"
            "    train_dataset=tokenized_ds_train,\n"
            "    eval_dataset=tokenized_ds_val,\n"
            "    compute_metrics=compute_metrics,\n"
            ")\n\n"
            "trainer_lora_distilmBERT.train()\n",
        ),
        md(
            "## 6. Evaluate on the Validation Split\n\n"
            "These metrics tell me whether the adapter-tuned model is strong enough to keep as a deployment candidate."
        ),
        old[14],
        old[15],
        md(
            "## 7. Log the Model to MLflow\n\n"
            "The final training step stores the tokenizer, adapter weights, and run metadata in MLflow so I can compare experiments cleanly later."
        ),
        old[17],
    ]
    save_notebook(path, nb)


def clean_04():
    path, nb = load_notebook("04_miniLM_trainning.ipynb")
    if nb["cells"] and first_line(nb["cells"][0]) == "# MiniLM LoRA Training":
        return
    if nb["cells"] and first_line(nb["cells"][0]) == "# MiniLM QLoRA Training":
        nb["cells"][0]["source"] = [
            "# MiniLM LoRA Training\n",
            "\n",
            "This notebook fine-tunes `microsoft/Multilingual-MiniLM-L12-H384` for scam detection. "
            "The target is a model with high scam recall that is still small enough for inexpensive cloud deployment.",
        ]
        save_notebook(path, nb)
        return
    old = nb["cells"]
    nb["cells"] = [
        md(
            "# MiniLM LoRA Training\n\n"
            "This notebook fine-tunes `microsoft/Multilingual-MiniLM-L12-H384` for scam detection. "
            "The target is a model with high scam recall that is still small enough for inexpensive cloud deployment."
        ),
        md(
            "## 1. Import Training Dependencies\n\n"
            "The stack combines Hugging Face Transformers, PEFT, and MLflow so the training run is lightweight, traceable, and easy to compare against other baselines."
        ),
        set_source(
            old[2],
            "# Training, PEFT, and logging dependencies\n\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "import numpy as np\n\n"
            "from transformers import (\n"
            "    AutoModelForSequenceClassification,\n"
            "    AutoTokenizer,\n"
            "    TrainingArguments,\n"
            "    Trainer,\n"
            "    EarlyStoppingCallback,\n"
            ")\n"
            "from peft import LoraConfig, get_peft_model\n"
            "import tensorflow as tf\n"
            "import evaluate\n"
            "import mlflow\n"
            "import mlflow.transformers\n"
            "from mlflow.tracking import MlflowClient\n"
            "from datasets import Dataset, Value, Sequence, Features\n",
        ),
        md(
            "## 2. Load the Prepared Dataset\n\n"
            "I train from the balanced Hugging Face dataset produced earlier so this run stays aligned with the rest of the experiment pipeline."
        ),
        set_source(
            old[4],
            "# Load the balanced training split generated during data preparation\n"
            "training_set_path = '../data/processed/train_ds/data-00000-of-00001.arrow'\n"
            "ds = Dataset.from_file(training_set_path)\n"
            "ds.to_pandas().head(10)\n",
        ),
        set_source(
            old[5],
            "# Create a validation split for checkpoint selection and metric tracking\n"
            "ds_split = ds.train_test_split(test_size=0.1, seed=42)\n\n"
            "ds_train = ds_split['train']\n"
            "ds_val = ds_split['test']\n\n"
            "print(f\"Train Set: {len(ds_train)}, {ds_train.to_pandas()['labels'].value_counts()}\")\n"
            "print(f\"Validation Set: {len(ds_val)}, {ds_val.to_pandas()['labels'].value_counts()}\")\n",
        ),
        md(
            "## 3. Select the Base Model\n\n"
            "MiniLM is a good candidate for cloud deployment because it is materially smaller than larger transformer encoders while still strong on multilingual classification."
        ),
        set_source(old[7], "minilm_id = 'microsoft/Multilingual-MiniLM-L12-H384'\n"),
        md(
            "## 4. Encode Labels and Tokenize Inputs\n\n"
            "I convert labels into integers and tokenize with a fixed 256-token limit. "
            "That truncation length is a practical compromise between scam-message coverage and inference cost."
        ),
        set_source(
            old[9],
            "# Convert the text labels into numeric targets and tokenize the messages\n"
            "str2int = {'fraud': 1, 'ham': 0}\n\n"
            "encoded_ds_train = ds_train.map(lambda x: {'labels': str2int[x['labels']]})\n"
            "encoded_ds_val = ds_val.map(lambda x: {'labels': str2int[x['labels']]})\n\n"
            "tokenizer = AutoTokenizer.from_pretrained(minilm_id)\n\n"
            "def tokenize_function(examples):\n"
            "    return tokenizer(\n"
            "        examples['text'],\n"
            "        padding='max_length',\n"
            "        truncation=True,\n"
            "        max_length=256,\n"
            "    )\n\n"
            "tokenized_ds_train = encoded_ds_train.map(tokenize_function, batched=True)\n"
            "tokenized_ds_val = encoded_ds_val.map(tokenize_function, batched=True)\n\n"
            "print(tokenized_ds_train.features)\n"
            "print(f'Train Set:{tokenized_ds_train}\\n')\n"
            "print(tokenized_ds_val.features)\n"
            "print(f'Validation Set:{tokenized_ds_val}\\n')\n",
        ),
        md(
            "## 5. Configure LoRA Fine-Tuning\n\n"
            "The base model stays mostly frozen while trainable low-rank adapters learn the task. "
            "This reduces memory pressure and makes local fine-tuning practical on Apple Silicon."
        ),
        set_source(
            old[11],
            "from scipy.special import softmax\n"
            "from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score\n"
            "import numpy as np\n\n"
            "# Load the base classifier in reduced precision to keep the run lightweight\n"
            "minilm_model = AutoModelForSequenceClassification.from_pretrained(\n"
            "    minilm_id,\n"
            "    num_labels=2,\n"
            "    torch_dtype='float16',\n"
            ")\n\n"
            "lora_config = LoraConfig(\n"
            "    r=8,\n"
            "    lora_alpha=16,\n"
            "    bias='none',\n"
            "    lora_dropout=0.1,\n"
            "    task_type='SEQ_CLS',\n"
            "    target_modules=['query', 'key', 'value', 'dense'],\n"
            ")\n\n"
            "minilm_lora = get_peft_model(minilm_model, lora_config)\n\n"
            "def compute_metrics(eval_pred):\n"
            "    logits, labels = eval_pred\n"
            "    predicted_class = np.argmax(logits, axis=-1)\n"
            "    return {\n"
            "        'accuracy': accuracy_score(labels, predicted_class),\n"
            "        'precision': precision_score(labels, predicted_class),\n"
            "        'recall': recall_score(labels, predicted_class),\n"
            "        'f1_score': f1_score(labels, predicted_class),\n"
            "    }\n\n"
            "training_args = TrainingArguments(\n"
            "    output_dir='./results_lora_minilm',\n"
            "    num_train_epochs=4,\n"
            "    per_device_train_batch_size=4,\n"
            "    per_device_eval_batch_size=16,\n"
            "    gradient_accumulation_steps=4,\n"
            "    weight_decay=0.01,\n"
            "    save_strategy='best',\n"
            "    greater_is_better=True,\n"
            "    load_best_model_at_end=True,\n"
            "    metric_for_best_model='recall',\n"
            "    fp16=False,\n"
            "    bf16=True,\n"
            ")\n",
        ),
        md(
            "## 6. Train and Evaluate\n\n"
            "Recall is the checkpoint selection target because missing a scam is more costly than sending a few extra alerts for human review."
        ),
        set_source(
            old[13],
            "from torch import mps\n\n"
            "tf.random.set_seed(42)\n"
            "mps.empty_cache()\n\n"
            "trainer_minilm_lora = Trainer(\n"
            "    model=minilm_lora,\n"
            "    args=training_args,\n"
            "    train_dataset=tokenized_ds_train,\n"
            "    eval_dataset=tokenized_ds_val,\n"
            "    compute_metrics=compute_metrics,\n"
            ")\n\n"
            "trainer_minilm_lora.train()\n",
        ),
        old[14],
        md(
            "## 7. Register the Run in MLflow\n\n"
            "Logging the final adapter and metrics makes the training run reproducible and supports later deployment selection."
        ),
        old[16],
    ]
    save_notebook(path, nb)


def clean_eval(name: str, model_uri: str, model_label: str):
    path, nb = load_notebook(name)
    expected_title = f"# {model_label} Evaluation on Unseen Data"
    if nb["cells"] and first_line(nb["cells"][0]) == expected_title:
        return
    old = nb["cells"]
    model_var = "distilmbert_model" if "distil" in name.lower() else "minilm_model"
    nb["cells"] = [
        md(
            f"# {model_label} Evaluation on Unseen Data\n\n"
            "This notebook measures the final holdout performance of the trained model. "
            "I evaluate classification quality, latency, throughput, and model footprint because deployment cost matters alongside recall."
        ),
        md(
            "## 1. Import Evaluation Dependencies\n\n"
            "The evaluation stack loads the model from MLflow, tokenizes the holdout set, and computes both business-facing and systems-facing metrics."
        ),
        set_source(
            old[2],
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from datasets import Dataset\n\n"
            "import mlflow\n"
            "import mlflow.transformers\n\n"
            "import numpy as np\n"
            "import torch\n"
            "from torch.utils.data import DataLoader\n"
            "from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score\n",
        ),
        md(
            "## 2. Load the Holdout Test Set\n\n"
            "This dataset was kept separate from training so the reported metrics better reflect generalization to unseen scam messages."
        ),
        set_source(
            old[4],
            "# Load the unseen holdout split created during data preparation\n"
            "test_ds = Dataset.from_file('../data/processed/test_ds/data-00000-of-00001.arrow')\n"
            "test_ds.to_pandas()['labels'].value_counts()\n",
        ),
        md(
            "## 3. Load the Trained Model from MLflow\n\n"
            "Using the registered MLflow artifact keeps evaluation aligned with the exact model version that would be deployed."
        ),
        set_source(
            old[7],
            "mlflow.set_tracking_uri('http://localhost:8080')\n"
            f"model_run_id = '{model_uri}'\n\n"
            "components = mlflow.transformers.load_model(model_run_id)\n"
            f"{model_var} = components.model\n"
            "tokenizer = components.tokenizer\n",
        ),
        md(
            "## 4. Encode Labels and Tokenize the Holdout Set\n\n"
            "The holdout split is tokenized with the same preprocessing recipe used during training so the comparison stays fair."
        ),
        set_source(
            old[9],
            "labels2id = {'fraud': 1, 'ham': 0}\n"
            "encoded_test_ds = test_ds.map(lambda x: {'labels': labels2id[x['labels']]})\n\n"
            "tokenized_ds = encoded_test_ds.map(\n"
            "    lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=256),\n"
            "    batched=True,\n"
            ")\n\n"
            "tokenized_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])\n\n"
            "print(tokenized_ds.features)\n"
            "print(tokenized_ds)\n",
        ),
        md(
            "## 5. Measure Inference Latency and Throughput\n\n"
            "These measurements help determine whether the model is realistic for low-cost cloud serving, not just whether it scores well on classification metrics."
        ),
        set_source(
            old[11],
            "import time\n\n"
            "device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')\n"
            f"{model_var}.to(device)\n"
            f"{model_var}.eval()\n\n"
            "loader = DataLoader(tokenized_ds, batch_size=4)\n"
            "predictions = []\n"
            "latencies = []\n\n"
            "with torch.no_grad():\n"
            "    for batch in loader:\n"
            "        input_ids = batch['input_ids'].to(device)\n"
            "        attention_mask = batch['attention_mask'].to(device)\n\n"
            "        start_time = time.time()\n"
            f"        outputs = {model_var}(input_ids=input_ids, attention_mask=attention_mask)\n"
            "        logits = outputs.logits\n"
            "        batch_predictions = torch.argmax(logits, dim=-1)\n"
            "        end_time = time.time()\n\n"
            "        predictions.extend(batch_predictions.cpu().tolist())\n"
            "        latencies.append((end_time - start_time) * 1000)\n\n"
            "throughput = len(test_ds) / (np.sum(latencies) / 1000)\n\n"
            "inference_metrics = {\n"
            "    'p50_latency_ms': float(np.percentile(latencies, 50)),\n"
            "    'p95_latency_ms': float(np.percentile(latencies, 95)),\n"
            "    'p99_latency_ms': float(np.percentile(latencies, 99)),\n"
            "    'throughput_req_sec': float(throughput),\n"
            "}\n",
        ),
        old[12],
        md(
            "## 6. Inspect Model Footprint\n\n"
            "Parameter count and memory footprint are used as deployment proxies when comparing candidates for a cloud endpoint."
        ),
        set_source(
            old[14],
            f"num_params = {model_var}.num_parameters()\n"
            f"memory_size = {model_var}.get_memory_footprint() / (1024**2)\n"
            f"width = {model_var}.config.hidden_size\n"
            f"length = {model_var}.config.num_hidden_layers\n\n"
            "architecture_metrics = {\n"
            "    'num_parameters': num_params,\n"
            "    'model_width': width,\n"
            "    'model_length': length,\n"
            "    'memory_size_MB': memory_size,\n"
            "}\n\n"
            "pd.DataFrame(architecture_metrics, index=[0])\n",
        ),
        md(
            "## 7. Compute Classification Metrics\n\n"
            "Accuracy alone is not enough for scam detection, so I track precision, recall, and F1 to understand the false-positive / false-negative trade-off."
        ),
        set_source(
            old[18],
            "pred_vs_actual = {\n"
            "    'predictions': np.int8(predictions),\n"
            "    'actual': np.int8(tokenized_ds['labels']),\n"
            "}\n"
            "pred_vs_actual = pd.DataFrame(pred_vs_actual)\n",
        ),
        old[19],
        md(
            "## 8. Visualize Error Structure\n\n"
            "The confusion matrix shows where the classifier still misses scams or over-flags legitimate messages."
        ),
        set_source(
            old[21],
            "from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay\n\n"
            "cm = confusion_matrix(\n"
            "    y_true=pred_vs_actual['actual'],\n"
            "    y_pred=pred_vs_actual['predictions'],\n"
            "    normalize='true',\n"
            ")\n\n"
            "disp = ConfusionMatrixDisplay(cm)\n"
            "disp.plot(cmap=plt.cm.Blues, values_format='.1%')\n"
            "print(cm)\n"
            "plt.show()\n",
        ),
        md(
            "## 9. Count Misclassifications\n\n"
            "A quick error count gives a simple summary of how many examples still need investigation for future data or prompt improvements."
        ),
        set_source(
            old[23],
            "false_pred = pred_vs_actual[pred_vs_actual['predictions'] != pred_vs_actual['actual']]\n"
            "false_pred.value_counts()\n",
        ),
    ]
    save_notebook(path, nb)


def main():
    clean_01()
    clean_02()
    clean_03()
    clean_04()
    clean_eval(
        "05_miniLM_evaluation.ipynb",
        "models:/m-0ec65a1c22bc4ab1ada2f5ddb7980a40",
        "MiniLM",
    )
    clean_eval(
        "06_distilmBERT_evaluation.ipynb",
        "models:/m-1b441e092a3f43869dea8d3abf3e4ad8",
        "Distil mBERT",
    )


if __name__ == "__main__":
    main()
