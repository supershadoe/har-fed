import logging
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from target_class import TargetClass
from models.base_paper_cnn import BasePaperCNN
from train.dataset import Pamap2Dataset

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(asctime)s: %(message)s',
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "../dataset"
SUBJECTS_TO_USE = list(range(101, 109))
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

DATALOADER_WORKERS = 64

def main():
    logger.info(f"Using device: {DEVICE}")

    all_train_dfs, all_test_dfs = [], []
    used_acts = [act.value for act in ACTIVITIES_TO_USE]
    for subject_id in SUBJECTS_TO_USE:
        logging.debug(f"[subject{subject_id}] Creating 80/20 split")
        df = pd.read_parquet(
            f"{DATA_DIR}/subject{subject_id}.parquet.zst"
        )
        df = df[
            df['activity_id'].isin(used_acts)
        ]

        train_subject_df, test_subject_df = train_test_split(
            df, test_size=0.20, shuffle=False
        )
        all_train_dfs.append(train_subject_df)
        all_test_dfs.append(test_subject_df)

    logging.debug(f"Merge all train and test dfs")
    train_df = pd.concat(all_train_dfs, ignore_index=True)
    test_df = pd.concat(all_test_dfs, ignore_index=True)
    logger.info(f"Total training samples: {len(train_df)}, Test samples: {len(test_df)}")

    feature_cols = [col for col in train_df.columns if col not in ['timestamp', 'activity_id', 'subject_id']]
    logger.debug(
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

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=DATALOADER_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=DATALOADER_WORKERS)

    logging.debug(f"Initialize model")
    model = BasePaperCNN(n_features=len(feature_cols), n_classes=len(ACTIVITIES_TO_USE)).to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=0.9,
    )
    criterion = nn.CrossEntropyLoss()
    
    logger.info("Starting model training...")
    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    logger.info("Evaluating final model...")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluating"):
            outputs = model(X_batch.to(DEVICE))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    final_accuracy = accuracy_score(all_labels, all_preds)
    logger.info(f"Final Centralized Accuracy: {final_accuracy * 100:.2f}%")
    logger.info("--- Detailed Classification Report ---")

    print(classification_report(
        all_labels, all_preds,
        labels=list(range(len(ACTIVITIES_TO_USE))),
        target_names=[act.name for act in ACTIVITIES_TO_USE],
    ))

if __name__ == '__main__':
    main()
