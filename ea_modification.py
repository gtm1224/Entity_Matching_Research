import pandas as pd
import numpy as np
import re
import random

# simple English stopword list (you can swap in NLTK/spacy if you want)
STOPWORDS = {
    "a","an","the","and","or","but","if","then","else","when","while","to","of","in","on","at","for","from","by",
    "with","without","as","is","are","was","were","be","been","being","it","this","that","these","those",
    "i","you","he","she","we","they","me","him","her","us","them","my","your","his","its","our","their",
    "not","no","so","too","very","can","could","should","would","will","just","do","does","did","doing"
}

_word_re = re.compile(r"\w+|[^\w\s]")  # keeps punctuation as separate tokens

def shorten_explanation(row, col="explanation", stopwords=STOPWORDS, seed_col=None):
    text = row[col]
    if pd.isna(text) or not str(text).strip():
        return text

    tokens = _word_re.findall(str(text))
    orig_len = len(tokens)
    target_len = max(1, orig_len // 2)   # reduce total length by half

    # RNG (optionally deterministic per row)
    rnd = random.Random(row[seed_col]) if seed_col is not None else random

    # 1) remove stopwords (word tokens only)
    filtered = []
    for t in tokens:
        if re.fullmatch(r"\w+", t):  # word
            if t.lower() in stopwords:
                continue
        filtered.append(t)

    # 2) randomly drop tokens until length == target_len (keep order)
    if len(filtered) > target_len:
        keep_idx = sorted(rnd.sample(range(len(filtered)), target_len))
        filtered = [filtered[i] for i in keep_idx]

    # join with nice spacing: no space before punctuation
    out = []
    for t in filtered:
        if out and re.fullmatch(r"[^\w\s]", t):
            out[-1] += t
        else:
            out.append(t)
    return " ".join(out)

def apply_shorten(df):
    df["explanation_short"]=df.apply(shorten_explanation, axis=1)


if __name__ == "__main__":

    df = pd.read_csv("./data/ea_data/abt-buy_train_ea.csv")
    apply_shorten(df)
    print(df.head(10))
    df.to_csv("test_ea_shorten.csv", index=False)