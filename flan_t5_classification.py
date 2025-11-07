import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from utils import *
import pandas as pd
import os, glob

def load_trained_t5(model_dir,device=None):

    """
    Loads a fine-tuned T5 model and tokenizer for inference.

    Args:
        model_dir (str): Path to the saved model directory.
        device (str, optional): "cuda", "cpu", or "auto".
                                If None, it picks automatically.

    Returns:
        model (T5ForConditionalGeneration): Loaded model in eval mode.
        tokenizer (T5TokenizerFast): Matching tokenizer.
        device (torch.device): The actual device used.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)
    tokenizer = T5TokenizerFast.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    print(f"✅ Model loaded on {device}")
    return model, tokenizer, device

# only output for one sentence
def predict_label(text, tokenizer, model,device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=8)
    pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return pred.strip().lower()

def norm_label(s: str) -> str:
    s = s.strip().lower()
    return "no match" if s.startswith("no") else "match"


@torch.no_grad()
def evaluate_dataset(model, tokenizer, texts, gold_labels=None, device=None,
                     batch_size=32, max_in=256, max_out=4):
    """
    Args:
        model, tokenizer:  loaded trained T5 from sotred path
        texts: list of serialized "[entity_a] ...\n[entity_b] ..." strings
               (or a pandas Series, or HF Dataset column)
        gold_labels: list/Series of "match"/"no match" (optional, for metrics)
        device: torch.device or None (auto-detect)
        batch_size: generation batch size
        max_in, max_out: encoder input cap / decoder max_new_tokens

    Returns:
        preds_norm: list[str] of normalized predictions ("match"/"no match")
        metrics (dict) if gold_labels provided, else None
    """
    if device is None:
        device = next(model.parameters()).device

    if hasattr(texts, "tolist"):
        texts = texts.tolist()
    if gold_labels is not None and hasattr(gold_labels, "tolist"):
        gold_labels = gold_labels.tolist()

    preds = []
    for i in range(0,len(texts),batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", truncation=True,
                        padding=True, max_length=max_in).to(device)
        outs = model.generate(**enc, max_new_tokens=max_out)
        batch_preds = tokenizer.batch_decode(outs, skip_special_tokens=True)
        preds.extend(batch_preds)
    preds_norm = [norm_label(p) for p in preds]
    metrics = None
    if gold_labels is not None:
        gold_norm = [norm_label(g) for g in gold_labels] #the true label
        prec, rec, f1, _ = precision_recall_fscore_support(
            gold_norm, preds_norm, average="binary", pos_label="match")
        acc = accuracy_score(gold_norm, preds_norm)
        metrics = {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}

    return preds_norm, metrics


if __name__ == '__main__':
    base_dir = "./flan_t5_abtbuy_with_ea"
    ckpts = sorted(glob.glob(os.path.join(base_dir, "checkpoint-*")),
               key=lambda p: int(p.rsplit("-", 1)[-1]))
    model_dir = ckpts[-1]  # automatically pick latest checkpoint

    print(f"Loading model from {model_dir}")
    model,tokenizer,device = load_trained_t5(model_dir)

    beer_test = pd.read_csv("./data/Beer/test.csv")
    tableA = pd.read_csv("./data/Beer/tableA.csv").fillna("")
    tableB = pd.read_csv("./data/Beer/tableB.csv").fillna("")

    # preprocess dataframes → add 'sample' and 'new_label'
    preprocessing_dataset_auto(beer_test, tableA, tableB)
    texts = beer_test["sample"].tolist()
    gold_labels = beer_test["new_label"].tolist()
    preds_norm,metrics = evaluate_dataset(model,tokenizer, texts, gold_labels)
    print(metrics)
