import argparse
import pathlib
import sys

import pandas as pd
from tqdm.auto import tqdm
from logs import LOGGER

ALL_FEATURES = [
    'timestamp', 'activity_id', 'heart_rate',
    *(
        f'{i}_{j}'
        for i in ('hand', 'chest', 'ankle')
        for j in (
            'temp_c', # dropped
            'acc16_x', 'acc16_y', 'acc16_z', # kept
            'acc6_x', 'acc6_y', 'acc6_z', # dropped
            'gyr_x', 'gyr_y', 'gyr_z', # kept
            'mgt_x', 'mgt_y', 'mgt_z', # dropped
            'orient_x', 'orient_y', 'orient_z', 'orient_w', # dropped
        )
    ),
]
FEATURES_TO_DROP = [
    f'{i}_{j}'
    for i in ('hand', 'chest', 'ankle')
    for j in (
        'temp_c',
        'acc6_x', 'acc6_y', 'acc6_z',
        'mgt_x', 'mgt_y', 'mgt_z',
        'orient_x', 'orient_y', 'orient_z', 'orient_w',
    )
]

def preprocess_pamap2(dataset_path: pathlib.Path, output_path: pathlib.Path):
    for i in tqdm(range(101, 110), desc="Processing Subjects", colour="cyan"):
        file_path = dataset_path / "Protocol" / f"subject{i}.dat"
        optional_file_path = dataset_path / "Optional" / f"subject{i}.dat"

        dfs: list[pd.DataFrame] = [
            pd.read_csv(file_path, sep=" ", header=None, names=ALL_FEATURES)
        ]
        if optional_file_path.exists():
            dfs.append(pd.read_csv(
                optional_file_path,
                sep=" ",
                header=None,
                names=ALL_FEATURES,
            ))

        subject_df = pd.concat(dfs, ignore_index=True)
        subject_df.drop(FEATURES_TO_DROP, axis=1, inplace=True)
        subject_df = subject_df.ffill()
        subject_df = subject_df[subject_df['activity_id'] != 0].reset_index(drop=True)
        subject_df['subject_id'] = i
        subject_df.dropna(inplace=True)

        output_filename = output_path / f"subject{i}.parquet.zst"
        subject_df.to_parquet(output_filename, compression='zstd')

    LOGGER.info(
        f"Preprocessing complete; Saved processed files at {output_path}.",
    )


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
        LOGGER.fatal('Dataset not found; Run download_dataset.sh first!')
        sys.exit(127)

    preprocess_pamap2(dataset_path, output_path)
