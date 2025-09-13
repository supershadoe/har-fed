import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from logs import LOGGER
from models.lstm import HarLstmModel
from target_class import TargetClass
from train.dataset import Pamap2Dataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSED_DATA_DIR = "../dataset"
SUBJECTS_TO_USE = list(range(101, 109))
RARE_ACTIVITIES = [9, 11, 20] # watching TV, car driving, playing soccer

# Hyperparameters
WINDOW_SIZE = 256
STEP = 128
BATCH_SIZE = 512
EPOCHS = 15
LEARNING_RATE = 0.001
DATALOADER_WORKERS = 64


def main():
    """Main function to run the LOSO cross-validation."""
    LOGGER.info(f"Using device: {DEVICE}")

    temp_df = pd.read_parquet(f"{PROCESSED_DATA_DIR}/subject101.parquet.zst")
    feature_cols = [col for col in temp_df.columns if col not in ['timestamp', 'activity_id']]

    label_map = {}
    i = 0
    for act in TargetClass:
        if act.value in RARE_ACTIVITIES:
            continue
        label_map[act.value] = i
        i += 1
    n_classes = len(label_map)

    final_scores = []

    LOGGER.info("Starting...")

    for test_subject_id in tqdm(SUBJECTS_TO_USE, desc="LOSO Folds"):
        logging.debug(f"[subject{test_subject_id}] Concating train set")
        train_subject_ids = [sid for sid in SUBJECTS_TO_USE if sid != test_subject_id]
        train_dfs = [pd.read_parquet(f"{PROCESSED_DATA_DIR}/subject{sid}.parquet.zst") for sid in train_subject_ids]
        train_df = pd.concat(train_dfs, ignore_index=True)

        logging.debug(f"[subject{test_subject_id}] Creating test set")
        test_df = pd.read_parquet(f"{PROCESSED_DATA_DIR}/subject{test_subject_id}.parquet.zst")

        logging.debug(f"[subject{test_subject_id}] Drop rare acts")
        train_df = train_df[~train_df['activity_id'].isin(RARE_ACTIVITIES)]
        test_df = test_df[~test_df['activity_id'].isin(RARE_ACTIVITIES)]

        logging.debug(f"[subject{test_subject_id}] identify activity blocks")
        train_df['block_id'] = ((train_df['subject_id'] != train_df['subject_id'].shift()) | (train_df['activity_id'] != train_df['activity_id'].shift())).cumsum()
        test_df['block_id'] = ((test_df['subject_id'] != test_df['subject_id'].shift()) | (test_df['activity_id'] != test_df['activity_id'].shift())).cumsum()

        logging.debug(f"[subject{test_subject_id}] normalize data")
        scaler = StandardScaler()
        train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
        test_df[feature_cols] = scaler.transform(test_df[feature_cols])

        logging.debug(f"[subject{test_subject_id}] convert from pd to np ndarray")
        X_train_np = train_df[feature_cols].to_numpy()
        y_train_np = train_df['activity_id'].to_numpy()
        block_ids_train = train_df['block_id'].to_numpy()

        X_test_np = test_df[feature_cols].to_numpy()
        y_test_np = test_df['activity_id'].to_numpy()
        block_ids_test = test_df['block_id'].to_numpy()
        
        logging.debug(f"[subject{test_subject_id}] create dataset and dataloaders")
        train_dataset = Pamap2Dataset(X_train_np, y_train_np, block_ids_train, label_map, WINDOW_SIZE, STEP)
        test_dataset = Pamap2Dataset(X_test_np, y_test_np, block_ids_test, label_map, WINDOW_SIZE, STEP)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=DATALOADER_WORKERS)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=DATALOADER_WORKERS)

        logging.debug(f"[subject{test_subject_id}] Initialize CNN")
        model = HarLstmModel(n_features=len(feature_cols), n_classes=n_classes).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
            logging.debug(f"[subject{test_subject_id}] Start training epoch {epoch}")
            model.train()
            for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Training", leave=False):
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in tqdm(test_loader, desc="Evaluating", leave=False):
                outputs = model(X_batch.to(DEVICE))
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.numpy())

        score = f1_score(all_labels, all_preds, average='weighted')
        final_scores.append(score)
        LOGGER.info(f"FOLD SCORE for Subject {test_subject_id} (Weighted F1): {score:.4f}")

    LOGGER.info("Cross-Validation Complete.")
    LOGGER.info(f"Scores from all folds: {[f'{s:.4f}' for s in final_scores]}")
    LOGGER.info(f"Average Weighted F1 Score: {np.mean(final_scores):.4f}")
    LOGGER.info(f"Standard Deviation: {np.std(final_scores):.4f}")


if __name__ == '__main__':
    main()
