from pathlib import Path
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, ConfusionMatrixDisplay

# ========= CONFIGURAZIONE =========

MODEL_NAME = "Musixmatch/umberto-commoncrawl-cased-v1"
OUT = Path("results/umberto_finetuning")
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "train": "data/subset/train-validation/train_subset.csv",
    "validation": "data/subset/train-validation/validation_subset.csv",
    "test": "data/subset/test/labeled_test_subset.csv",
}


# ========= DATI =========

def load_split(path):
    # Legge il CSV e corregge eventuali spazi nei nomi colonna, es. "text ".
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    return Dataset.from_pandas(df, preserve_index=False)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    # Tokenizzazione dei documenti per UmBERTo.
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)


train_ds = load_split(FILES["train"]).map(tokenize, batched=True).rename_column("label", "labels")
valid_ds = load_split(FILES["validation"]).map(tokenize, batched=True).rename_column("label", "labels")
test_ds = load_split(FILES["test"]).map(tokenize, batched=True).rename_column("label", "labels")

cols = ["input_ids", "attention_mask", "labels"]
train_ds.set_format("torch", columns=cols)
valid_ds.set_format("torch", columns=cols)
test_ds.set_format("torch", columns=cols)


# ========= MODELLO E METRICHE =========

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)


def compute_metrics(eval_pred):
    # Calcola accuracy, F1 e ROC-AUC sulle predizioni del modello.
    logits, labels = eval_pred
    z = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "roc_auc": roc_auc_score(labels, probs[:, 1]),
    }


# Compatibilità con versioni diverse di transformers: eval_strategy/evaluation_strategy.
args = {
    "output_dir": str(OUT / "checkpoints"),
    "num_train_epochs": 3,
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "logging_strategy": "epoch",
    "save_strategy": "epoch",
    "report_to": "none",
    "load_best_model_at_end": False,
}
args["eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"] = "epoch"

trainer = Trainer(
    model=model,
    args=TrainingArguments(**args),
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# ========= TRAINING: 3 EPOCHE CON VALIDATION A OGNI EPOCA =========

trainer.train()
history = pd.DataFrame(trainer.state.log_history)
history.to_csv(OUT / "training_history.csv", index=False)

valid_metrics = history[history["eval_loss"].notna()].copy()
valid_metrics.to_csv(OUT / "validation_metrics_by_epoch.csv", index=False)


# ========= TEST: SOLO DOPO LA TERZA EPOCA =========

test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
pd.DataFrame([test_metrics]).to_csv(OUT / "test_metrics_after_epoch_3.csv", index=False)

pred = trainer.predict(test_ds)
y_true = pred.label_ids
y_pred = pred.predictions.argmax(axis=1)


# ========= GRAFICI =========

# Curva loss training/validation.
train_loss = history[history["loss"].notna()][["epoch", "loss"]]
valid_loss = history[history["eval_loss"].notna()][["epoch", "eval_loss"]]

plt.figure(figsize=(8, 5))
plt.plot(train_loss["epoch"], train_loss["loss"], marker="o", label="Training loss")
plt.plot(valid_loss["epoch"], valid_loss["eval_loss"], marker="o", label="Validation loss")
plt.xlabel("Epoca")
plt.ylabel("Loss")
plt.title("UmBERTo - Training e validation loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "umberto_loss_curve.png", dpi=300)
plt.close()

# Metriche sul validation set dopo ogni epoca.
plt.figure(figsize=(8, 5))
for col, name in [("eval_accuracy", "Accuracy"), ("eval_f1", "F1"), ("eval_roc_auc", "ROC-AUC")]:
    plt.plot(valid_metrics["epoch"], valid_metrics[col], marker="o", label=name)
plt.xlabel("Epoca")
plt.ylabel("Score")
plt.title("UmBERTo - Metriche validation per epoca")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "umberto_validation_metrics.png", dpi=300)
plt.close()

# Matrice di confusione sul test finale.
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["umano", "macchina"])
plt.title("UmBERTo - Matrice di confusione test")
plt.tight_layout()
plt.savefig(OUT / "umberto_test_confusion_matrix.png", dpi=300)
plt.close()


# ========= SALVATAGGIO MODELLO FINALE =========

trainer.save_model(OUT / "final_model")
tokenizer.save_pretrained(OUT / "final_model")

print(f"Fine. Risultati salvati in: {OUT}")
