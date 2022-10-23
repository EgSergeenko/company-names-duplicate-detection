import logging
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine
from sklearn.metrics import f1_score


def get_logger(
    level=logging.INFO,
    fmt='[%(asctime)s] %(message)s',
):
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter(fmt=fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def load_data(
    data_dir,
    data_filename,
):
    data = pd.read_csv(
        os.path.join(
            data_dir,
            data_filename,
        ),
        delimiter=';',
    )
    sequences = data.iloc[:, 0].tolist()
    labels = data.iloc[:, 1].tolist()
    augmentation = data.iloc[:, 2].tolist()
    return sequences, labels, augmentation


def get_data_split(
    sequences,
    labels,
    augmentations,
    val_size=0.05,
    test_size=0.05,
):
    val_test_labels_set = random.sample(
        list(set(labels)),
        round(len(set(labels)) * (test_size + val_size)),
    )
    split = len(val_test_labels_set) // 2
    val_labels_set = set(val_test_labels_set[:split])
    test_labels_set = set(val_test_labels_set[split:])

    train_sequences, train_labels = [], []
    val_sequences, val_labels = [], []
    test_sequences, test_labels = [], []
    for sequence, label, augmentation in zip(sequences, labels, augmentations):
        if label in test_labels_set:
            if not augmentation:
                test_sequences.append(sequence)
                test_labels.append(label)
        elif label in val_labels_set:
            if not augmentation:
                val_sequences.append(sequence)
                val_labels.append(label)
        else:
            train_sequences.append(sequence)
            train_labels.append(label)
    return (
        (train_sequences, train_labels),
        (val_sequences, val_labels),
        (test_sequences, test_labels),
    )


def get_embeddings(
    model,
    data_loader,
    device,
):
    embeddings, labels = [], []
    model.eval()
    for sample in data_loader:
        encoding, label = sample[0]
        with torch.no_grad():
            h_0, c_0 = model.init_hidden()
            h_0, c_0 = h_0.to(device), c_0.to(device)
            output = model(encoding.to(device), h_0, c_0)
            embeddings.append(output.cpu().numpy())
            labels.append(label.item())
    return embeddings, labels


def get_distances(
    embeddings,
):
    distances = pd.DataFrame(
        data=np.zeros((len(embeddings), len(embeddings))),
    )
    for i, embedding in enumerate(embeddings):
        for j, anchor in enumerate(embeddings):
            if i > j:
                distance = cosine(embedding, anchor)
                distances.iloc[i, j] = distance
    return distances


def get_query_anchor_split(
    embeddings,
    labels,
):
    labels_cnt = Counter(labels)
    anchor_labels_set = {lbl for lbl, cnt in labels_cnt.items() if cnt >= 2}

    query_embeddings, query_labels, query_indexes = [], [], []
    anchor_embeddings, anchor_labels, anchor_indexes = [], [], []
    indexes = np.arange(0, len(embeddings), dtype=int)
    for embedding, label, idx in zip(embeddings, labels, indexes):
        if label in anchor_labels_set:
            anchor_embeddings.append(embedding)
            anchor_labels.append(label)
            anchor_indexes.append(idx)
            anchor_labels_set.remove(label)
        else:
            query_embeddings.append(embedding)
            query_labels.append(label)
            query_indexes.append(idx)
    return (
        (query_embeddings, query_labels, query_indexes),
        (anchor_embeddings, anchor_labels, anchor_indexes),
    )


def get_predictions(
    distances,
    query_indexes,
    anchor_indexes,
    threshold,
):
    y_pred = []
    for query_idx in query_indexes:
        predicted_label = 0
        for anchor_idx in anchor_indexes:
            if query_idx > anchor_idx:
                distance = distances.iloc[query_idx, anchor_idx]
            else:
                distance = distances.iloc[anchor_idx, query_idx]
            if distance < threshold:
                predicted_label = 1
                break
        y_pred.append(predicted_label)
    return y_pred


def evaluate_f1_score(embeddings, labels, best_threshold=None):
    distances = get_distances(embeddings)

    queries, anchors = get_query_anchor_split(embeddings, labels)
    query_embeddings, query_labels, query_indexes = queries
    anchor_embeddings, anchor_labels, anchor_indexes = anchors

    y_true = []
    for query_label in query_labels:
        if query_label in anchor_labels:
            y_true.append(1)
        else:
            y_true.append(0)

    if best_threshold is None:
        min_distance = np.amin(distances.values)
        max_distance = np.amax(distances.values)
        thresholds = np.linspace(min_distance, max_distance, 100)
    else:
        thresholds = [best_threshold]
    scores = []
    for threshold in thresholds:
        y_pred = get_predictions(
            distances,
            query_indexes,
            anchor_indexes,
            threshold,
        )
        score = f1_score(y_true, y_pred)
        scores.append((score, threshold))
    return scores
