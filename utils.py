import pandas as pd
import numpy as np
MAX_IN, MAX_OUT = 256, 4

def fmt(x):
    s = str(x).strip()
    return s if s and s.lower() != "nan" else "[EMPTY]"



def serialize_pair_auto(row, tableA, tableB):
    """
    Automatically serialize any dataset pair using all columns except 'id'.
    Works for Beer, Abt-Buy, Amazon-Google, etc.
    """
    # Retrieve rows by ID
    left  = tableA.loc[tableA["id"] == row["ltable_id"]].iloc[0]
    right = tableB.loc[tableB["id"] == row["rtable_id"]].iloc[0]

    # Get all usable column names (ignore 'id')
    cols = [c for c in tableA.columns if c.lower() != "id"]

    # Helper to format a single entity
    def make_entity(e):
        parts = []
        for col in cols:
            val = fmt(e[col])  # your existing fmt() handles NaN
            parts.append(f"[{col.upper()}] {val}")
        return " ".join(parts)

    A = make_entity(left)
    B = make_entity(right)

    return f"[entity_a] {A}\n[entity_b] {B}"


def serialize_pair_beer(row, tableA, tableB):
    left  = tableA.loc[tableA["id"] == row["ltable_id"]].iloc[0]
    right = tableB.loc[tableB["id"] == row["rtable_id"]].iloc[0]
    A = f'[Beer_NAME] {fmt(left["Beer_Name"])} [Brew_Factory_Name] {fmt(left["Brew_Factory_Name"])} [Style] {fmt(left["Style"])} [ABV] {fmt(left["ABV"])}'
    B = f'[Beer_NAME] {fmt(right["Beer_Name"])} [Brew_Factory_Name] {fmt(right["Brew_Factory_Name"])} [Style] {fmt(right["Style"])} [ABV] {fmt(right["ABV"])}'
    return f"[entity_a] {A}\n[entity_b] {B}"



def apply_entity_auto(df, tableA, tableB):
    df["sample"] = df.apply(lambda row: serialize_pair_auto(row, tableA, tableB), axis=1)

def match_no_match(row):
    label = None
    if row["label"]==1:
        label = "match"
    else:
        label = "no match"

    if "explanation" in row:
        label += f' [explanation] {fmt(row["explanation"])}'

    return label

def apply_label(df):
    df["new_label"] = df.apply(match_no_match, axis=1)

def preprocessing_dataset_auto(df, tableA, tableB):
    apply_label(df)
    apply_entity_auto(df, tableA, tableB)


MAX_IN, MAX_OUT = 256, 128

def preprocess_fn(tokenizer, batch):
    enc = tokenizer(batch["input"], truncation=True, padding="max_length", max_length=MAX_IN)
    with tokenizer.as_target_tokenizer():
        dec = tokenizer(batch["label"], truncation=True, padding="max_length", max_length=MAX_OUT)
    enc["labels"] = dec["input_ids"]
    return enc

def norm_label(s: str) -> str:
    s = s.strip().lower()
    return "no match" if s.startswith("no") else "match"

# def make_compute_metrics(tokenizer):
#     def compute_metrics(eval_pred):
#         preds, labels = eval_pred
#
#         # Some HF versions return (preds, labels); sometimes preds is a tuple
#         if isinstance(preds, tuple):
#             preds = preds[0]
#
#         preds = np.asarray(preds)
#
#         # If preds are logits (B, T, V), convert to token IDs with argmax
#         if preds.ndim == 3:
#             pred_ids = preds.argmax(axis=-1)
#         else:
#             # Already token IDs (B, T)
#             pred_ids = preds
#
#         # Replace -100 in labels for decoding
#         label_ids = np.where(labels != -100, labels, tokenizer.pad_token_id)
#
#         pred_strs = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#         gold_strs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
#
#         y_pred = [norm_label(p) for p in pred_strs]
#         y_true = [norm_label(g) for g in gold_strs]
#
#         prec, rec, f1, _ = precision_recall_fscore_support(
#             y_true, y_pred, average="binary", pos_label="match"
#         )
#         acc = accuracy_score(y_true, y_pred)
#         return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}
#
#     return compute_metrics
# utils.py

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def make_compute_metrics(tokenizer):
    pad_id = tokenizer.pad_token_id
    eos_id = getattr(tokenizer, "eos_token_id", None)
    vocab_size = getattr(tokenizer, "vocab_size", None)

    def _to_py_obj(arr):
        # unwrap tuple -> first element
        if isinstance(arr, tuple):
            arr = arr[0]
        # keep raggedness
        return np.array(arr, dtype=object).tolist()

    def _clean_ids(row):
        # row may be list/np.array/tuple; coerce to list of python ints
        seq = [int(x) for x in list(row)]
        # if eos exists, truncate at first eos (inclusive) to reduce junk tails
        if eos_id is not None and eos_id in seq:
            seq = seq[: seq.index(eos_id) + 1]
        # drop any invalid ids (negative or >= vocab_size if known)
        if vocab_size is not None:
            seq = [t for t in seq if 0 <= t < vocab_size]
        else:
            seq = [t for t in seq if t >= 0]
        # if ended up empty, put a pad
        if not seq:
            seq = [pad_id]
        return seq

    def _replace_ignore(row):
        # replace -100 with pad for label decoding
        return [tid if tid != -100 else pad_id for tid in list(row)]

    def compute_metrics(eval_pred):
        preds, labels = eval_pred

        pred_rows  = _to_py_obj(preds)
        label_rows = _to_py_obj(labels)

        pred_ids  = [_clean_ids(r) for r in pred_rows]
        label_ids = [_replace_ignore(r) for r in label_rows]

        pred_strs  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        gold_strs  = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        def norm_label(s: str) -> str:
            s = s.strip().lower()
            return "no match" if s.startswith("no") else "match"

        y_pred = [norm_label(p) for p in pred_strs]
        y_true = [norm_label(g) for g in gold_strs]

        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label="match")
        acc = accuracy_score(y_true, y_pred)
        return {"precision": float(p), "recall": float(r), "f1": float(f1), "accuracy": float(acc)}

    return compute_metrics
