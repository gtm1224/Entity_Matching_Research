import pandas as pd



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
    return "match" if row["label"] == 1 else "no match"

def apply_label(df):
    df["new_label"] = df.apply(match_no_match, axis=1)

def preprocessing_dataset_auto(df, tableA, tableB):
    apply_label(df)
    apply_entity_auto(df, tableA, tableB)