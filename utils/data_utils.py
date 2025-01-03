import librosa
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def load_wave(filename, resample=True, sr=32000):
    input, sample_rate = librosa.load(filename, sr=sr, mono=True)
    if (resample) & (sample_rate != sr):
        input = librosa.resample(input, sample_rate, sr, res_type="kaiser_fast")
    return input


def normalize_wave(y):
    return librosa.util.normalize(y)

def crop_or_pad_range(y, length, start):
    if len(y) < length:
        y = np.concatenate([y, np.zeros(length - len(y))])

        n_repeats = length // len(y)
        epsilon = length % len(y)

        y = np.concatenate([y] * n_repeats + [y[:epsilon]])

    elif len(y) > length:
        y = y[start:start + length]

    return y

def load_and_crop_range(filename, period = 5, resample=True, sr=32000, normalize=True, start = 0):
    wave = load_wave(filename, resample=resample, sr=sr)
    length = period * sr
    wave = crop_or_pad_range(wave, length, start)
    if normalize:
        return normalize_wave(wave)
    return wave

def prepare_df_secondary_labels(df):
    df["secondary_labels"] = (
        df["secondary_labels"]
        .map(
            lambda s: s.replace("[", "")
            .replace("]", "")
            .replace(",", "")
            .replace("'", "")
            .split(" ")
        ).values
    )
    return df

def prepare_df_config(df, data_config):
    target_columns = sorted(df["primary_label"].unique())
    num_classes = len(target_columns)
    bird2id = {b: i for i, b in enumerate(target_columns)}
    data_config["num_classes"] = num_classes
    data_config["bird2id"] = bird2id
    data_config["target_columns"] = target_columns
    return data_config

def prepare_df_year(df):
    x = df["path"].str.split("\\", expand=True)
    x = x[x.columns[1]]
    x = x.str.split("/", expand=True)
    x = x[x.columns[0]]
    x = x.str.split("-", expand=True)
    x = x[x.columns[1]]
    return x.astype("int")


def prepare_year_data_split(year_df, KFOLD=5, seed=2020):
    values = year_df["primary_label"].value_counts()
    labels, counts = values.index, values.values
    valid_mask = counts < 2
    labels = labels[valid_mask]
    additional_samples = []
    for label in labels:
        additional_sample = year_df[year_df["primary_label"] == label].sample(n=1)
        additional_samples.append(additional_sample)
    if len(additional_samples) > 0:
        additional_samples = pd.concat(additional_samples)
        year_df = pd.concat([year_df, additional_samples]).reset_index(drop=True)

    skf = StratifiedKFold(n_splits=KFOLD, random_state=seed, shuffle=True)
    year_df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=year_df, y=year_df["primary_label"].values)):
        year_df.loc[val_idx, 'fold'] = fold

    return year_df



def prepare_whole_data_split(df, KFOLD=5, seed=2020):
    year_dfs = []
    for year in df["year"].unique():
        year_df = df[df["year"] == year].reset_index(drop=True)
        values = year_df["primary_label"].value_counts()
        labels, counts = values.index, values.values
        valid_mask = counts < 2
        labels = labels[valid_mask]
        additional_samples = []
        for label in labels:
            additional_sample = year_df[year_df["primary_label"] == label].sample(n=1)
            additional_samples.append(additional_sample)
        if len(additional_samples) > 0:
            additional_samples = pd.concat(additional_samples)
            year_df = pd.concat([year_df, additional_samples]).reset_index(drop=True)

        skf = StratifiedKFold(n_splits=KFOLD, random_state=seed, shuffle=True)
        year_df['fold'] = -1
        for fold, (train_idx, val_idx) in enumerate(skf.split(X=year_df, y=year_df["primary_label"].values)):
            year_df.loc[val_idx, 'fold'] = fold

        year_dfs.append(year_df)

    df = pd.concat(year_dfs).reset_index(drop=True)
    return df

import os
def setup_output_dir(config):
    os.makedirs(config["output_folder"], exist_ok=True)
    exp_folder = os.path.join(config["output_folder"], config["exp_name"])
    os.makedirs(exp_folder, exist_ok=True)
    config["exp_folder"] = exp_folder
    return config

def normalize_rating(df):
    return np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)


def do_kfold(df, KFOLD=5, seed=42):
    skf = StratifiedKFold(n_splits=KFOLD, random_state=seed, shuffle=True)
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(X=df, y=df["primary_label"].values)):
        df.loc[val_idx, 'fold'] = fold
    return df


def read_dataframe():
    df = pd.read_csv('../data/birdclef-2024/train_metadata.csv')
    df["path"] = "../data/birdclef-2024/train_audio/" + df["filename"]
    return df


def prepare_teacher_pred(target_columns):
    kaggle_model_predictions = pd.read_csv('../data/birdclef-2024/train_meta_data_pseudo_2.csv')
    return kaggle_model_predictions[target_columns].values.tolist()

