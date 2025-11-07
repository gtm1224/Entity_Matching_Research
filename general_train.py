from utils import *
from transformers import (
    T5ForConditionalGeneration, T5TokenizerFast,
    DataCollatorForSeq2Seq, TrainingArguments, Trainer,Seq2SeqTrainingArguments, Seq2SeqTrainer
)
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch
import gc

if __name__ == "__main__":
    if torch.cuda.is_available():
        gc.collect(); torch.cuda.empty_cache()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print("✅ Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ CUDA not available; running on CPU.")
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    train_dir= "./data/ea_data/abt-buy_train_ea.csv"
    valid_dir="./data/abt-buy/valid.csv"
    test_dir= "./data/abt-buy/test.csv"
    tableA_dir = "./data/abt-buy/tableA.csv"
    tableB_dir = "./data/abt-buy/tableB.csv"

    pairs_train = pd.read_csv(train_dir)
    pairs_valid = pd.read_csv(valid_dir)
    pairs_test = pd.read_csv(test_dir)
    tableA = pd.read_csv(tableA_dir).fillna("")
    tableB = pd.read_csv(tableB_dir).fillna("")

    preprocessing_dataset_auto(pairs_train, tableA, tableB)
    preprocessing_dataset_auto(pairs_valid, tableA, tableB)
    preprocessing_dataset_auto(pairs_test, tableA, tableB)


    print(pairs_train.head(10))
    ds = DatasetDict({
        "train": Dataset.from_dict({
            "input": pairs_train["sample"].tolist(),
            "label": pairs_train["new_label"].tolist()
        }),
        "validation": Dataset.from_dict({
            "input": pairs_valid["sample"].tolist(),
            "label": pairs_valid["new_label"].tolist()
        }),
        "test": Dataset.from_dict({
            "input": pairs_test["sample"].tolist(),
            "label": pairs_test["new_label"].tolist()
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

    # model.config.use_cache = False
    # training config
    args = Seq2SeqTrainingArguments(
        output_dir="flan_t5_abtbuy_with_ea",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.05,
        logging_steps=100,
        eval_strategy="no",
        save_strategy="epoch",
        load_best_model_at_end=False,
        metric_for_best_model="f1",
        fp16=False,  # or False if CPU
        bf16=True,
        predict_with_generate=True,           # <<< key: no logits kept
        generation_max_length=128,
        generation_num_beams=1,
        eval_accumulation_steps=32,
    )

    # trainer
    trainer = Seq2SeqTrainer(
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
        enc = tokenizer(text,return_tensors="pt", truncation=True, max_length=MAX_IN).to(model.device)
        out = model.generate(**enc, max_new_tokens=MAX_OUT)
        return norm_label(tokenizer.decode(out[0], skip_special_tokens=True))

    print("Example pred:", predict_label(ds["test"][0]["input"]))

