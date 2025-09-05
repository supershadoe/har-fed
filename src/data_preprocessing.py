import argparse
import logging
import pathlib
import sys
from typing import List

import pandas as pd
from tqdm import tqdm

from annotations import Attributes

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s: %(message)s',
)
logger = logging.getLogger(__name__)

def preprocess_pamap2(dataset_path: pathlib.Path, output_path: pathlib.Path):
    """
    Loads raw PAMAP2 data for each subject, cleans it, and saves each
    subject's data to a separate CSV file.
    """
    attributes = Attributes(is_preprocessing=True)

    for i in tqdm(range(101, 110), desc="Processing Subjects"):
        file_path = dataset_path / "Protocol" / f"subject{i}.dat"
        optional_file_path = dataset_path / "Optional" / f"subject{i}.dat"

        dfs: List[pd.DataFrame] = [
            pd.read_csv(file_path, sep=" ", header=None, names=attributes.to_keep)
        ]
        if optional_file_path.exists():
            dfs.append(pd.read_csv(
                optional_file_path,
                sep=" ",
                header=None,
                names=attributes.to_keep,
            ))

        subject_df = pd.concat(dfs, ignore_index=True)
        subject_df.drop(attributes.to_drop, axis=1, inplace=True)
        subject_df = subject_df[subject_df['activity_id'] != 0].reset_index(drop=True)
        subject_df['heart_rate'] = subject_df['heart_rate'].ffill()
        subject_df.dropna(inplace=True)
        subject_df['subject_id'] = i

        output_filename = output_path / f"subject{i}.csv"
        subject_df.to_csv(output_filename, index=False)

    logger.info("Preprocessing complete. Individual files saved.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data preprocessing script for PAMAP2 dataset',
    )
    parser.add_argument(
        '-p',
        '--path',
        default='PAMAP2_Dataset',
        help='Location of the unzipped PAMAP2 dataset',
    )
    parser.add_argument(
        '-o',
        '--output-path',
        default='dataset',
        help='Location for storing processed datasets',
    )
    args = parser.parse_args()
    
    dataset_path = pathlib.Path(args.path)
    output_path = pathlib.Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        logger.fatal('Dataset not found; Run download_dataset.sh first!')
        sys.exit(127)

    preprocess_pamap2(dataset_path, output_path)
