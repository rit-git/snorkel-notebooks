import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import train_test_split


def load_youtube_dataset(load_train_labels: bool = True, split_dev: bool = True, delimiter: str=None):
    filenames = sorted(glob.glob("data/Youtube*.csv"))

    dfs = []
    for i, filename in enumerate(filenames, start=1):
        df = pd.read_csv(filename)
        # Lowercase column names
        df.columns = map(str.lower, df.columns)
        # Remove comment_id field
        df = df.drop("comment_id", axis=1)
        # Add field indicating source video
        df["video"] = [i] * len(df)
        # Rename fields
        df = df.rename(columns={"class": "label", "content": "text"})
        # Remove delimiter chars
        df['text'].replace(regex=True, inplace=True, to_replace=delimiter, value=r'')
        # Shuffle order
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        dfs.append(df)

    df_train = pd.concat(dfs[:4])
    if split_dev:
        df_dev = df_train.sample(500, random_state=123)

    if not load_train_labels:
        df_train["label"] = np.ones(len(df_train["label"])) * -1
    df_valid_test = dfs[4]
    df_valid, df_test = train_test_split(
        df_valid_test, test_size=250, random_state=123, stratify=df_valid_test.label
    )

    if split_dev:
        return df_train, df_dev, df_valid, df_test
    else:
        return df_train, df_valid, df_test


def load_amazon_dataset(load_train_labels: bool = False, split_dev: bool = True, delimiter: str=None):
    filenames = sorted(glob.glob("data/Amazon*.csv"))

    dfs = []
    for i, filename in enumerate(filenames, start=1):
        df = pd.read_csv(filename, header=None)
        # Lowercase column names
        df.columns = ["key", "text", "label"]
        # Remove delimiter chars
        if delimiter:
            df['text'].replace(regex=True, inplace=True, to_replace=delimiter, value=r'')
        # Shuffle order
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)
        dfs.append(df)

    df_train = dfs[1]
    if split_dev:
        df_dev = df_train.sample(500, random_state=123)

    if not load_train_labels:
        df_train["label"] = np.ones(len(df_train["label"])) * -1
    df_valid_test = dfs[0]
    df_valid, df_test = train_test_split(
        df_valid_test, test_size=250, random_state=123, stratify=df_valid_test.label
    )

    # sample from training set to keep to a reasonable size
    df_train = df_train.sample(5000, random_state=123)

    if split_dev:
        return df_train, df_dev, df_valid, df_test
    else:
        return df_train, df_valid, df_test

if __name__=="__main__":
    df_train, df_dev, df_valid, df_test = load_amazon_dataset()