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
    print("input: ",text)
    print("pred: ",pred)
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
        # print("batch_preds: ",batch_preds)
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


def get_text_gold_labels_for_evaluation(data,tableA,tableB):
    preprocessing_dataset_auto(data, tableA, tableB)
    texts = data["sample"].tolist()
    gold_labels = data["new_label"].tolist()
    return texts, gold_labels

def evaluate_dataset_wrapper(model, tokenizer, data, tableA, tableB):
    texts,gold_labels = get_text_gold_labels_for_evaluation(data,tableA,tableB)
    preds_norm, metrics = evaluate_dataset(model, tokenizer, texts, gold_labels)
    print(metrics)


if __name__ == '__main__':
    base_dir = "./flan_t5_abtbuy_with_ea"
    ckpts = sorted(glob.glob(os.path.join(base_dir, "checkpoint-*")),
               key=lambda p: int(p.rsplit("-", 1)[-1]))
    model_dir = ckpts[-1]  # automatically pick latest checkpoint

    print(f"Loading model from {model_dir}")
    model,tokenizer,device = load_trained_t5(model_dir)

    train_dir, valid_dir, test_dir, tableA_dir, tableB_dir, output_dir = get_dir_for_base_model_training("Beer")
    # print(train_dir, valid_dir, test_dir, tableA_dir, tableB_dir)
    # exit()
    pairs_train = pd.read_csv(train_dir)
    pairs_valid = pd.read_csv(valid_dir)
    pairs_test = pd.read_csv(test_dir)
    tableA = pd.read_csv(tableA_dir).fillna("")
    tableB = pd.read_csv(tableB_dir).fillna("")

    # evaluate using all the train, valid,and test datasets, print out the results, take the minimum F1
    # preprocess dataframes → add 'sample' and 'new_label'
    # preprocessing_dataset_auto(beer_test, tableA, tableB)
    # texts = beer_test["sample"].tolist()
    # gold_labels = beer_test["new_label"].tolist()
    # texts,gold_labels = get_text_gold_labels_for_evaluation(pairs_train,tableA,tableB)
    # preds_norm,metrics = evaluate_dataset(model,tokenizer, texts, gold_labels)
    # print(metrics)
    #
    # texts, gold_labels = get_text_gold_labels_for_evaluation(pairs_train, tableA, tableB)
    # preds_norm, metrics = evaluate_dataset(model, tokenizer, texts, gold_labels)
    # print(metrics)
    #
    # texts, gold_labels = get_text_gold_labels_for_evaluation(pairs_train, tableA, tableB)
    # preds_norm, metrics = evaluate_dataset(model, tokenizer, texts, gold_labels)
    # print(metrics)
    total_text = []
    total_gold_labels = []
    texts_train,gold_labels_train = get_text_gold_labels_for_evaluation(pairs_train,tableA,tableB)
    total_text.extend(texts_train)
    total_gold_labels.extend(gold_labels_train)
    texts_valid, gold_labels_valid = get_text_gold_labels_for_evaluation(pairs_valid, tableA, tableB)
    total_text.extend(texts_valid)
    total_gold_labels.extend(gold_labels_valid)
    texts_test, gold_labels_test = get_text_gold_labels_for_evaluation(pairs_test, tableA, tableB)
    total_text.extend(texts_test)
    total_gold_labels.extend(gold_labels_test)

    preds_norm, metrics = evaluate_dataset(model, tokenizer, total_text, total_gold_labels)
    print("evaluated on total samples: \n")
    print(metrics)


    # evaluation on train, valid, and test separately
    print("evaluating on train samples: \n")
    evaluate_dataset_wrapper(model, tokenizer,pairs_train,tableA, tableB)
    print("evaluating on valid samples: \n")
    evaluate_dataset_wrapper(model, tokenizer, pairs_valid, tableA, tableB)
    print("evaluating on test samples: \n")
    evaluate_dataset_wrapper(model, tokenizer, pairs_test, tableA, tableB)



