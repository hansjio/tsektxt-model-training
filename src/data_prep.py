import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_clean(raw_path):
    df = pd.read_csv(raw_path)
    df = df[df["Label"] != "Label"]
    df["Label"] = df["Label"].astype(int)
    df = df.dropna(subset=["Content"]).drop_duplicates(subset="Content")
    df = df.reset_index(drop=True)
    return df


def split_and_save(df, processed_dir, seed=42):
    os.makedirs(processed_dir, exist_ok=True)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["Label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["Label"], random_state=seed
    )

    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)
    return train_df, val_df, test_df


def get_splits(raw_path, processed_dir, seed=42):
    """Load from processed cache if it exists, else build it fresh.
    All paths passed in should already be absolute (resolved by the caller),
    so this function has no dependency on the current working directory.
    """
    train_path = os.path.join(processed_dir, "train.csv")
    val_path = os.path.join(processed_dir, "val.csv")
    test_path = os.path.join(processed_dir, "test.csv")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        return (
            pd.read_csv(train_path),
            pd.read_csv(val_path),
            pd.read_csv(test_path),
        )

    df = load_and_clean(raw_path)
    return split_and_save(df, processed_dir, seed)