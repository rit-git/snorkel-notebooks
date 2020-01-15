import pandas as pd
import glob
import numpy as np
from sklearn.model_selection import train_test_split


def load_youtube_dataset(load_train_labels: bool = False, split_dev: bool = True, delimiter: str=None):
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
        df = df.reset_index(drop=True)
        dfs.append(df)
    
    # concatenate DFs, shuffle order
    df_full = pd.concat(dfs).sample(frac=1, random_state=123)
    train_size, dev_size, valid_size, test_size = 1000, 250, 200, 500
    assert(test_size+valid_size+dev_size+train_size <= len(df_full))
    if not split_dev:
        dev_size = 0
        
    df_test = df_full[:test_size]
    df_valid = df_full[test_size:test_size+valid_size]
    df_dev = df_full[test_size+valid_size:test_size+valid_size+dev_size]
    df_train = df_full[test_size+valid_size+dev_size:test_size+valid_size+dev_size+train_size]
    if not load_train_labels:
        df_train.drop("label", axis=1, inplace=True)

    assert(len(df_train) > 0)
    if split_dev:
        return df_train, df_dev, df_valid, df_test
    else:
        return df_train, df_valid, df_test


def load_amazon_dataset(load_train_labels: bool = False, split_dev: bool = True, delimiter: str=None):
    filenames = sorted(glob.glob("data/Amazon*Dev.csv"))

    dfs = []
    for i, filename in enumerate(filenames, start=1):
        df = pd.read_csv(filename, header=None)
        # Lowercase column names
        df.columns = ["key", "text", "label"]
        # Remove delimiter chars
        if delimiter:
            df['text'].replace(regex=True, inplace=True, to_replace=delimiter, value=r'')
        df = df.reset_index(drop=True)
        dfs.append(df)
    # concatenate DFs, shuffle order
    df_full = pd.concat(dfs).sample(frac=1, random_state=123)

    train_size, dev_size, valid_size, test_size = 2000, 500, 500, 1000
    assert(test_size+valid_size+dev_size+train_size <= len(df_full))
    
    if not split_dev:
        dev_size = 0

    df_test = df_full[:test_size]
    df_valid = df_full[test_size:test_size+valid_size]
    df_dev = df_full[test_size+valid_size:test_size+valid_size+dev_size]
    df_train = df_full[test_size+valid_size+dev_size:test_size+valid_size+dev_size+train_size]
    if not load_train_labels:
        df_train.drop("label", axis=1, inplace=True)
    
    assert(len(df_train) > 0)

    if split_dev:
        return df_train, df_dev, df_valid, df_test
    else:
        return df_train, df_valid, df_test




def load_film_dataset(load_train_labels: bool = True, split_dev: bool = True, delimiter: str=None):
    filename = "data/wiki_movie_plots.csv"
    df = pd.read_csv(filename)
    df = df[["text", "label", "Genre", "Title"]]

    # Remove delimiter chars
    if delimiter:
        df['text'].replace(regex=True, inplace=True, to_replace=delimiter, value=r'')
    # Shuffle order
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    test_size = 1000
    valid_size = 500
    if split_dev:
        dev_size = 500
    else:
        dev_size = 0
    train_size = 1500

    df_test = df[:test_size]
    df_valid = df[test_size:test_size+valid_size]
    df_dev = df[test_size+valid_size:test_size+valid_size+dev_size]
    df_train = df[test_size+valid_size+dev_size:test_size+valid_size+dev_size+train_size]
    if not load_train_labels:
        df_train.drop("label", axis=1, inplace=True)

    if split_dev:
        return df_train, df_dev, df_valid, df_test
    else:
        return df_train, df_valid, df_test

if __name__=="__main__":
    df_train, df_dev, df_valid, df_test = load_amazon_dataset()