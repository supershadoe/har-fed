import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flwr.client import Client, NumPyClient
from flwr.common import Context
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from logs import LOGGER
from models.base_paper_cnn import BasePaperCNN
from train.dataset import Pamap2Dataset
from target_class import TargetClass


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "../dataset"
ACTIVITIES_TO_USE = [
    TargetClass.LYING,
    TargetClass.SITTING,
    TargetClass.STANDING,
    TargetClass.WALKING,
    TargetClass.ASCENDING,
    TargetClass.DESCENDING,
    TargetClass.VACUUM,
    TargetClass.IRONING,
]

# Hyperparameters from the paper
WINDOW_SIZE = 64  # Paper, Section IV-B: "partitioned into 64-unit windows"
STEP = 32         # Paper, Section IV-B: "50% overlap"
BATCH_SIZE = 64   # Paper, Section III C-3: "batch size of 64"
EPOCHS = 100      # Paper, Section VI-A: "trained up to 100 epochs"
LEARNING_RATE = 0.01 # Paper, Section III C-3: "learning rate n of of 0.01"

#DATALOADER_WORKERS = 8


def get_parameters(model: nn.Module):
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]):
    model.load_state_dict(
        OrderedDict(zip(
            model.state_dict().keys(),
            map(lambda x: torch.Tensor(x), parameters)
        )),
        strict=True,
    )


def train_one_epoch(model, train_loader, epochs: int, device: str):
    LOGGER.info("Starting model training...")
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.01, momentum=0.9,
    )
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total += y_batch.size(0)
            correct += (torch.max(outputs.data, 1)[1] == y_batch).sum().item()
        epoch_loss /= len(train_loader.dataset)
        epoch_acc = correct / total
        LOGGER.debug(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def test(model, test_loader, device: str):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    all_preds, all_labels = [], []
    loss = 0
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluating"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss += criterion(outputs, y_batch).item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    LOGGER.info(f"Final Centralized Accuracy: {accuracy * 100:.2f}%")
    LOGGER.debug("--- Detailed Classification Report ---")
    LOGGER.debug(classification_report(
        all_labels, all_preds,
        labels=list(range(len(ACTIVITIES_TO_USE))),
        target_names=[act.name for act in ACTIVITIES_TO_USE],
    ))
    loss /= len(test_loader.dataset)
    return loss, accuracy


class FlClient(NumPyClient):
    def __init__(self, subject_id, model, train_loader, test_loader, device):
        self.subject_id = subject_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def get_parameters(self, config):
        LOGGER.debug(f"[Client {self.subject_id}] get_parameters")
        return get_parameters(self.model)

    def fit(self, parameters, config):
        LOGGER.debug(f"[Client {self.subject_id}] fit, config: {config}")
        set_parameters(self.model, parameters)
        train_one_epoch(self.model, self.train_loader, epochs=1, device=self.device)
        return get_parameters(self.model), len(self.train_loader), {}

    def evaluate(self, parameters, config): # type: ignore
        print(f"[Client {self.subject_id}] evaluate, config: {config}")
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.test_loader, device=self.device)
        return float(loss), len(self.test_loader), {"accuracy": float(accuracy)}


def client_fn(context: Context) -> Client:
    used_acts = [act.value for act in ACTIVITIES_TO_USE]
    subject_id = 101 + int(context.node_config["partition-id"])
    logging.debug(f"[subject{subject_id}] Creating 80/20 split")
    df = pd.read_parquet(
        f"{DATA_DIR}/subject{subject_id}.parquet.zst"
    )
    df = df[
        df['activity_id'].isin(used_acts)
    ]

    train_df, test_df = train_test_split(
        df, test_size=0.20, shuffle=False
    )
    LOGGER.info(f"[subject{subject_id}] Training samples: {len(train_df)}, Test samples: {len(test_df)}")

    feature_cols = [col for col in train_df.columns if col not in ['timestamp', 'activity_id', 'subject_id']]
    LOGGER.debug(
        f"{len(feature_cols)} Features used: {', '.join(feature_cols)}",
    )
    label_map = {label: i for i, label in enumerate(used_acts)}

    logging.debug(f"Normalize train and test dfs")
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])

    logging.debug(f"Finding blocks of activities")
    train_df['block_id'] = (train_df['activity_id'] != train_df['activity_id'].shift()).cumsum()
    test_df['block_id'] = (test_df['activity_id'] != test_df['activity_id'].shift()).cumsum()

    logging.debug(f"Convert dfs to ndarrays")
    X_train_np = train_df[feature_cols].to_numpy()
    y_train_np = train_df['activity_id'].to_numpy()
    block_ids_train = train_df['block_id'].to_numpy()
    
    X_test_np = test_df[feature_cols].to_numpy()
    y_test_np = test_df['activity_id'].to_numpy()
    block_ids_test = test_df['block_id'].to_numpy()

    logging.debug(f"Initialize dataset and data loaders")
    train_dataset = Pamap2Dataset(X_train_np, y_train_np, block_ids_train, label_map, WINDOW_SIZE, STEP)
    test_dataset = Pamap2Dataset(X_test_np, y_test_np, block_ids_test, label_map, WINDOW_SIZE, STEP)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = BasePaperCNN(n_features=len(feature_cols), n_classes=len(ACTIVITIES_TO_USE)).to(DEVICE)

    return FlClient(
        subject_id, model, train_loader, test_loader, DEVICE
    ).to_client()
