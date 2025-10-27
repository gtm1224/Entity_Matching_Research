from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast,
    DataCollatorForSeq2Seq, TrainingArguments, Trainer
)
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
# --- serialization helpers ---
def fmt(x):
    s = str(x).strip()
    return s if s and s.lower() != "nan" else "[EMPTY]"

def serialize_pair(row, tableA, tableB):
    left  = tableA.loc[tableA["id"] == row["ltable_id"]].iloc[0]
    right = tableB.loc[tableB["id"] == row["rtable_id"]].iloc[0]
    A = f'[NAME] {fmt(left["name"])} [DESCRIPTION] {fmt(left["description"])} [PRICE] {fmt(left["price"])}'
    B = f'[NAME] {fmt(right["name"])} [DESCRIPTION] {fmt(right["description"])} [PRICE] {fmt(right["price"])}'
    return f"[entity_a] {A}\n[entity_b] {B}"

def match_no_match(row):
    return "match" if row["label"] == 1 else "no match"

def apply_label(df):
    df["new_label"] = df.apply(match_no_match, axis=1)

def apply_entity(df, tableA, tableB):
    df["sample"] = df.apply(lambda row: serialize_pair(row, tableA, tableB), axis=1)

def preprocessing_dataset(df, tableA, tableB):
    apply_label(df)
    apply_entity(df, tableA, tableB)

# --- tokenization / metrics ---
MAX_IN, MAX_OUT = 256, 4

def preprocess_fn(tokenizer, batch):
    enc = tokenizer(batch["input"], truncation=True, padding="max_length", max_length=MAX_IN)
    with tokenizer.as_target_tokenizer():
        dec = tokenizer(batch["label"], truncation=True, padding="max_length", max_length=MAX_OUT)
    enc["labels"] = dec["input_ids"]
    return enc

def norm_label(s: str) -> str:
    s = s.strip().lower()
    return "no match" if s.startswith("no") else "match"

def make_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        # Some HF versions return (preds, labels); sometimes preds is a tuple
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = np.asarray(preds)

        # If preds are logits (B, T, V), convert to token IDs with argmax
        if preds.ndim == 3:
            pred_ids = preds.argmax(axis=-1)
        else:
            # Already token IDs (B, T)
            pred_ids = preds

        # Replace -100 in labels for decoding
        label_ids = np.where(labels != -100, labels, tokenizer.pad_token_id)

        pred_strs = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        gold_strs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        y_pred = [norm_label(p) for p in pred_strs]
        y_true = [norm_label(g) for g in gold_strs]

        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label="match"
        )
        acc = accuracy_score(y_true, y_pred)
        return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}

    return compute_metrics


# --- main ---
if __name__ == '__main__':

    # GPU usage:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print("✅ Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ CUDA not available; running on CPU.")

    # load CSVs
    abt_pairs_train = pd.read_csv("./data/abt-buy/train.csv")
    abt_pairs_valid = pd.read_csv("./data/abt-buy/valid.csv")
    abt_pairs_test  = pd.read_csv("./data/abt-buy/test.csv")
    tableA = pd.read_csv("./data/abt-buy/tableA.csv").fillna("")
    tableB = pd.read_csv("./data/abt-buy/tableB.csv").fillna("")

    # preprocess dataframes → add 'sample' and 'new_label'
    preprocessing_dataset(abt_pairs_train, tableA, tableB)
    preprocessing_dataset(abt_pairs_valid, tableA, tableB)
    preprocessing_dataset(abt_pairs_test,  tableA, tableB)

    # print(abt_pairs_train.iloc[:10])
    # exit()
    # build HF datasets
    ds = DatasetDict({
        "train": Dataset.from_dict({
            "input": abt_pairs_train["sample"].tolist(),
            "label": abt_pairs_train["new_label"].tolist()
        }),
        "validation": Dataset.from_dict({
            "input": abt_pairs_valid["sample"].tolist(),
            "label": abt_pairs_valid["new_label"].tolist()
        }),
        "test": Dataset.from_dict({
            "input": abt_pairs_test["sample"].tolist(),
            "label": abt_pairs_test["new_label"].tolist()
        }),
    })

    # model + tokenizer (use T5-specific classes for clarity/speed)
    MODEL = "google/flan-t5-base"
    tokenizer = T5TokenizerFast.from_pretrained(MODEL)
    model = T5ForConditionalGeneration.from_pretrained(MODEL)

    # tokenize
    tokenized = ds.map(lambda batch: preprocess_fn(tokenizer, batch),
                       batched=True, remove_columns=["input", "label"])

    # data collator pads dynamically per batch and sets label padding to -100
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    print(tokenized["train"][0])
    exit()
    # training config
    args = TrainingArguments(
        output_dir="flan_t5_abtbuy_label_only",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        predict_with_generate=True,
        fp16=False,
    )

    # trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    trainer.train()

    # evaluate
    print("Validation:", trainer.evaluate(tokenized["validation"]))
    print("Test:", trainer.evaluate(tokenized["test"]))

    # quick inference helper
    def predict_label(text: str) -> str:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_IN).to(model.device)
        out = model.generate(**enc, max_new_tokens=MAX_OUT)
        return norm_label(tokenizer.decode(out[0], skip_special_tokens=True))

    print("Example pred:", predict_label(ds["test"][0]["input"]))
