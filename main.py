import os
import numpy as np
import pandas as pd
import torch
import sentencepiece

from datasets import Dataset
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)

# =========================
# Konfiguration
# =========================
MODEL_NAME = "deepset/gbert-base"
CSV_PATH = "data.csv"
TEXT_COL = "text"
LABEL_COLS = ["stress", "schlaf", "bewegung", "ernaehrung"]

OUTPUT_DIR = "gbert_multilabel_output"
MAX_LENGTH = 128
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Training
NUM_EPOCHS = 4
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
WEIGHT_DECAY = 0.01

# Schwellenwert für Multi-Label-Entscheidung
THRESHOLD = 0.5

set_seed(RANDOM_STATE)


# =========================
# Daten laden
# =========================
df = pd.read_csv(CSV_PATH)

required_cols = [TEXT_COL] + LABEL_COLS
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Fehlende Spalten in der CSV: {missing}")

# Nur notwendige Spalten behalten
df = df[required_cols].dropna(subset=[TEXT_COL]).copy()
df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()

# Leere Texte entfernen
df = df[df[TEXT_COL] != ""].copy()

# Labels als int casten
for col in LABEL_COLS:
    df[col] = df[col].astype(int)

print("Datensatzgröße:", len(df))
print("Beispielzeile:")
print(df.head(1))


# =========================
# Train/Test Split
# =========================
train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
test_ds = Dataset.from_pandas(test_df.reset_index(drop=True))


# =========================
# Tokenizer
# =========================
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,use_fast=False)
except Exception as e:
    raise RuntimeError(f"Tokenizer konnte nicht geladen werden: {e}")

print("Tokenizer erfolgreich geladen:", type(tokenizer))


# =========================
# Preprocessing
# =========================
def preprocess_function(examples):
    """

    :type examples: object
    """
    tokenized = tokenizer(examples[TEXT_COL], truncation=True, max_length=MAX_LENGTH, )

    labels = []
    for i in range(len(examples[TEXT_COL])):
        label_vec = [float(examples[col][i]) for col in LABEL_COLS]
        labels.append(label_vec)

    tokenized["labels"] = labels
    return tokenized


train_ds = train_ds.map(preprocess_function, batched=True)
test_ds = test_ds.map(preprocess_function, batched=True)

keep_cols = {"input_ids", "attention_mask", "labels"}
train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in keep_cols])

print("Features nach Tokenisierung:", train_ds.column_names)
print("Erster tokenisierter Eintrag:")
print(train_ds[0])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# =========================
# Modell
# =========================
id2label = {i: name.upper() for i, name in enumerate(LABEL_COLS)}
label2id = {name.upper(): i for i, name in enumerate(LABEL_COLS)}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(LABEL_COLS),
    id2label=id2label,
    label2id=label2id,
)

# Wichtig für Multi-Label-Klassifikation
model.config.problem_type = "multi_label_classification"


# =========================
# Metriken
# =========================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = sigmoid(logits)
    preds = (probs >= THRESHOLD).astype(int)

    return {
        "micro_f1": f1_score(labels, preds, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
        "micro_precision": precision_score(labels, preds, average="micro", zero_division=0),
        "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
        "micro_recall": recall_score(labels, preds, average="micro", zero_division=0),
        "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
    }


# =========================
# TrainingArguments
# =========================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    greater_is_better=True,
    save_total_limit=2,
    report_to="none",
    fp16=torch.cuda.is_available(),
)


# =========================
# Trainer
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# =========================
# Training
# =========================
trainer.train()


# =========================
# Evaluation
# =========================
eval_results = trainer.evaluate()

print("\nEvaluation:")
for k, v in eval_results.items():
    print(f"{k}: {v}")


# =========================
# Vorhersagen + Report
# =========================
pred_output = trainer.predict(test_ds)
logits = pred_output.predictions
true_labels = pred_output.label_ids

probs = sigmoid(logits)
pred_labels = (probs >= THRESHOLD).astype(int)

print("\nClassification report pro Label:")
print(
    classification_report(
        true_labels,
        pred_labels,
        target_names=LABEL_COLS,
        zero_division=0,
    )
)


# =========================
# Ergebnisse speichern
# =========================
os.makedirs(OUTPUT_DIR, exist_ok=True)

results_df = test_df.reset_index(drop=True).copy()

for i, col in enumerate(LABEL_COLS):
    results_df[f"{col}_prob"] = probs[:, i]
    results_df[f"{col}_pred"] = pred_labels[:, i]

results_path = os.path.join(OUTPUT_DIR, "test_predictions.csv")
results_df.to_csv(results_path, index=False)

# =========================
# Detaillierte Fehleranalyse auf Testdaten
# =========================
error_rows = []

for i in range(len(results_df)):
    text = results_df.iloc[i][TEXT_COL]

    true_vec = true_labels[i]
    pred_vec = pred_labels[i]
    prob_vec = probs[i]

    true_active = [LABEL_COLS[j] for j in range(len(LABEL_COLS)) if true_vec[j] == 1]
    pred_active = [LABEL_COLS[j] for j in range(len(LABEL_COLS)) if pred_vec[j] == 1]

    false_positives = [LABEL_COLS[j] for j in range(len(LABEL_COLS)) if pred_vec[j] == 1 and true_vec[j] == 0]
    false_negatives = [LABEL_COLS[j] for j in range(len(LABEL_COLS)) if pred_vec[j] == 0 and true_vec[j] == 1]

    # Nur fehlerhafte Beispiele speichern
    if false_positives or false_negatives:
        row = {
            "text": text,
            "true_labels": ", ".join(true_active),
            "predicted_labels": ", ".join(pred_active),
            "false_positives": ", ".join(false_positives),
            "false_negatives": ", ".join(false_negatives),
        }

        for j, label in enumerate(LABEL_COLS):
            row[f"{label}_prob"] = float(prob_vec[j])

        error_rows.append(row)

errors_df = pd.DataFrame(error_rows)
errors_path = os.path.join(OUTPUT_DIR, "error_analysis.csv")
errors_df.to_csv(errors_path, index=False)

print(f"\nFehleranalyse gespeichert in: {errors_path}")
print(f"Anzahl fehlerhafter Testbeispiele: {len(errors_df)}")

model_path = os.path.join(OUTPUT_DIR, "best_model")
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

# =========================
# Manuelle Beispieltexte analysieren
# =========================
MANUAL_EXAMPLES = [
    "ich esse und schlafe zu wenig",
    "abends einschlafen fällt mir schwer. Ich habe oft viel im Kopf",
    "Ich bin ständig in Eile und schaffe es kaum etwas ordentliches zu essen",
    "ich sitze nur am PC und bewege mich kaum",
    "viel Stress auf Arbeit",
    "ich schlafe schlecht",
]

def predict_texts(texts, threshold=THRESHOLD):
    model.eval()

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    device = model.device
    encodings = {k: v.to(device) for k, v in encodings.items()}

    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits.detach().cpu().numpy()

    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(int)

    results = []
    for i, text in enumerate(texts):
        row = {
            "text": text,
            "predicted_labels": [LABEL_COLS[j] for j in range(len(LABEL_COLS)) if preds[i][j] == 1],
        }
        for j, label in enumerate(LABEL_COLS):
            row[f"{label}_prob"] = float(probs[i][j])
            row[f"{label}_pred"] = int(preds[i][j])
        results.append(row)

    return pd.DataFrame(results)

manual_results_df = predict_texts(MANUAL_EXAMPLES, threshold=0.35)
manual_results_path = os.path.join(OUTPUT_DIR, "manual_examples_predictions.csv")
manual_results_df.to_csv(manual_results_path, index=False)

print("\nManuelle Beispiele:")
print(manual_results_df.to_string(index=False))
print(f"\nManuelle Beispielvorhersagen gespeichert in: {manual_results_path}")

print(f"\nModell gespeichert unter: {model_path}")
print(f"Vorhersagen gespeichert unter: {results_path}")