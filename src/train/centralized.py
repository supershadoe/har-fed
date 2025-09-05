import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.cnn import CNNModel
from train.dataset import Pamap2Dataset


logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s: %(message)s',
)
logger = logging.getLogger(__name__)

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


def create_windows(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    step: int
):
    """Creates sliding windows of sensor data from a DataFrame."""
    windows = []
    labels = []

    for _, group in df.groupby('subject_id'):
        for i in range(0, len(group) - window_size, step):
            feature_window = group[feature_cols].iloc[i:i + window_size].values
            
            # Use the mode of the activity_id in the window as the label
            label = stats.mode(group['activity_id'].iloc[i:i + window_size])[0][0]
            
            windows.append(feature_window)
            labels.append(label)
            
    return np.array(windows, dtype=np.float32), np.array(labels, dtype=np.int64)


def main():
    """Main function to run the LOSO cross-validation."""
    logger.info(f"Using device: {DEVICE}")

    temp_df = pd.read_csv(f"{PROCESSED_DATA_DIR}/subject101.csv")
    feature_cols = [col for col in temp_df.columns if col not in ['timestamp', 'activity_id']]

    all_possible_activities = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 18, 19, 24]
    label_map = {label: i for i, label in enumerate(all_possible_activities)}
    n_classes = len(all_possible_activities)

    final_scores = []

    logger.info("Starting...")

    for test_subject_id in tqdm(SUBJECTS_TO_USE, desc="LOSO Folds"):

        train_subject_ids = [sid for sid in SUBJECTS_TO_USE if sid != test_subject_id]
        train_dfs = [pd.read_csv(f"{PROCESSED_DATA_DIR}/subject{sid}.csv") for sid in train_subject_ids]
        train_df = pd.concat(train_dfs, ignore_index=True)
        train_df['subject_id'] = train_df['subject_id'].astype(int)

        test_df = pd.read_csv(f"{PROCESSED_DATA_DIR}/subject{test_subject_id}.csv")
        test_df['subject_id'] = test_df['subject_id'].astype(int)


        train_df = train_df[~train_df['activity_id'].isin(RARE_ACTIVITIES)]
        test_df = test_df[~test_df['activity_id'].isin(RARE_ACTIVITIES)]

        X_train, y_train_raw = create_windows(train_df, feature_cols, WINDOW_SIZE, STEP)
        X_test, y_test_raw = create_windows(test_df, feature_cols, WINDOW_SIZE, STEP)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

        y_train = np.array([label_map[label] for label in y_train_raw])
        y_test = np.array([label_map[label] for label in y_test_raw])

        train_loader = DataLoader(Pamap2Dataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(Pamap2Dataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

        model = CNNModel(n_features=len(feature_cols), n_classes=n_classes).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(EPOCHS):
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
        logger.info(f"FOLD SCORE for Subject {test_subject_id} (Weighted F1): {score:.4f}")

    logger.info("Cross-Validation Complete.")
    logger.info(f"Scores from all folds: {[f'{s:.4f}' for s in final_scores]}")
    logger.info(f"Average Weighted F1 Score: {np.mean(final_scores):.4f}")
    logger.info(f"Standard Deviation: {np.std(final_scores):.4f}")


if __name__ == '__main__':
    main()
