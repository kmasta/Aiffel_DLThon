import random
import os
import re
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except:
        pass

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_data(paths):
    # str이면 list로 래핑
    if isinstance(paths, str):
        paths = [paths]

    texts, labels = [], []
    for p in paths:
        df = pd.read_csv(p)
        texts.extend(df['clean_text'])
        labels.extend(df['label'])
    return texts, labels


def load_texts(paths):
    # str이면 list로 래핑
    if isinstance(paths, str):
        paths = [paths]

    texts = []
    for p in paths:
        df = pd.read_csv(p)
        texts.extend(df['clean_text'])
    return texts

def train_val_test_split(texts, labels, val_ratio, test_ratio, seed):
    if test_ratio <= 0:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts,
            labels,
            test_size=val_ratio,
            random_state=seed,
            shuffle=True,
            stratify=labels,
        )
        return train_texts, train_labels, val_texts, val_labels, [], []

    # 1) train+val vs test
    t1, test_texts, l1, test_labels = train_test_split(
        texts,
        labels,
        test_size=test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    # 2) train vs val (val_ratio를 (1-test_ratio)로 정규화)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        t1,
        l1,
        test_size=val_ratio/(1-test_ratio),
        random_state=seed,
        shuffle=True,
        stratify=l1,
    )

    return train_texts, train_labels, val_texts, val_labels, test_texts, test_labels

def encode_labels(labels, config=None, method=None):
    if method is None:
        method = config.get('label_encoding','int') if config else 'int'
    le = LabelEncoder()
    int_lbl = le.fit_transform(labels)
    if method == 'onehot':
        return np.eye(len(le.classes_))[int_lbl]
    return int_lbl

def save_submission(preds, filename):
    template_path = "data/original_data/submission.csv"
    submission_df = pd.read_csv(template_path)
    submission_df.columns = ["idx", "target"]
    submission_df["target"] = preds
    submission_df.to_csv(filename, index=False)

def save_test_result(test_texts, preds, test_labels, test_result_file):
    tr_df.columns = ["text", "pred", "label"]
    tr_df["text"] = test_texts
    tr_df["pred"] = preds
    tr_df["label"] = test_labels
    tr_df.to_csv(test_result_file, index=False)
