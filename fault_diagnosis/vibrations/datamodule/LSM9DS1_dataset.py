from pathlib import Path

import cbor2
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


dataset_distribution = {
    'train': [
        "clutch_2/current-load-0A",
        "clutch_2/current-load-2A",
        "clutch_2/current-load-4A",
        "clutch_2/current-load-6A",
        "clutch_2/misalignment-current-load-6A",
        "clutch_2/misalignment-current-load-4A",
        "clutch_2/misalignment-2-current-load-0A",
        "clutch_2/misalignment-3-current-load-2A",
        "clutch_2/rotor-1-current-load-0A-clutch-tightened",
        "clutch_2/rotor-1-current-load-2A",
        "clutch_2/rotor-3-current-load-4A",
        "clutch_2/rotor-3-current-load-6A",
        "clutch_2/rotor-6-current-load-0A",
        "clutch_1/start-up-current-load-0A",
        "clutch_1/current-load-6A",
        "clutch_1/misalignment-current-load-0A",
        "clutch_1/misalignment-current-load-6A",
    ],
    'valid': [
        "clutch_2/current-load-0A-2",
        "clutch_2/current-load-4A-2",
        "clutch_2/misalignment-current-load-2A",
        "clutch_2/misalignment-2-current-load-6A",
        "clutch_2/misalignment-3-current-load-0A",
        "clutch_2/misalignment-3-current-load-4A",
        "clutch_2/rotor-1-current-load-0A",
        "clutch_2/rotor-1-current-load-4A",
        "clutch_2/rotor-3-current-load-2A",
        "clutch_2/rotor-6-current-load-6A",
        "clutch_1/current-load-2A",
        "clutch_1/misalignment-current-load-2A",
    ],
    'test': [
        "clutch_2/current-load-2A-2",
        "clutch_2/current-load-6A-2",
        "clutch_2/misalignment-current-load-0A",
        "clutch_2/misalignment-2-current-load-2A",
        "clutch_2/misalignment-2-current-load-4A",
        "clutch_2/misalignment-3-current-load-6A",
        "clutch_2/rotor-1-current-load-6A",
        "clutch_2/rotor-3-current-load-0A",
        "clutch_2/rotor-6-current-load-2A",
        "clutch_2/rotor-6-current-load-4A",
        "clutch_1/start-up-current-load-0A-2",
        "clutch_1/current-load-4A",
        "clutch_1/misalignment-current-load-4A",
    ]
}


def extract_cbor(filepath: str) -> pd.DataFrame:
    with open(filepath, 'rb') as f:
        data = cbor2.decoder.load(f)
    
    df = pd.DataFrame(
        data['payload']['values'], 
        columns=['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ', 'magX', 'magY', 'magZ']
    )
    df['accX'] -= 9.81
    
    return df


def get_dataset_file_paths(dataset_path: Path, files_extension: str) -> list[Path]:
    files = dataset_path.rglob(f'*{files_extension}')
    files = sorted(files)
    return files


def prepare_data(dataset_files: list[Path], window_length: int, window_stride: int, classes: int = 3, normalize: bool = False):
    features_vectors, class_labels = [], []
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    for dataset_file in dataset_files:
        df = extract_cbor(dataset_file)

        steps = int((len(df) - window_length) / window_stride + 1)

        if classes == 3:
            if 'misalignment' in dataset_file.parent.name:
                label = 1
            elif 'rotor' in dataset_file.parent.name:
                label = 2
            else:
                label = 0
        elif classes == 7:
            if 'misalignment-3' in dataset_file.parent.name:
                label = 3
            elif 'misalignment-2' in dataset_file.parent.name:
                label = 2
            elif 'misalignment-current' in dataset_file.parent.name:
                label = 1
            elif 'rotor-1' in dataset_file.parent.name:
                label = 4
            elif 'rotor-3' in dataset_file.parent.name:
                label = 5
            elif 'rotor-6' in dataset_file.parent.name:
                label = 6
            else:
                label = 0

        for i in range(steps):
            a = df[['accX', 'accY', 'accZ']].iloc[i*window_stride:i*window_stride+window_length].values.flatten()

            if dataset_file.parents[1].name + '/' + dataset_file.parents[0].name in dataset_distribution['test']:
                X_test.append(a)
                y_test.append(label)
            elif dataset_file.parents[1].name + '/' + dataset_file.parents[0].name in dataset_distribution['valid']:
                X_val.append(a)
                y_val.append(label)
            elif dataset_file.parents[1].name + '/' + dataset_file.parents[0].name in dataset_distribution['train']:
                X_train.append(a)
                y_train.append(label)

            features_vectors.append(a)
            class_labels.append(label)   

    X_train = np.array(X_train)
    if normalize:
        scaler = MinMaxScaler(feature_range=(-1, 1)) 
        X_train = scaler.fit_transform(X_train)
    X_train = np.reshape(X_train, (-1, window_length, classes))
    y_train = np.array(y_train)

    X_val = np.array(X_val)
    if normalize:
        X_val = scaler.transform(X_val)
    X_val = np.reshape(X_val, (-1, window_length, classes))
    y_val = np.array(y_val)

    X_test = np.array(X_test)
    if normalize:
        X_test = scaler.transform(X_test)
    X_test = np.reshape(X_test, (-1, window_length, classes))
    y_test = np.array(y_test)

    features_vectors = np.array(features_vectors)
    if normalize:
        scaler = MinMaxScaler(feature_range=(-1, 1)) 
        features_vectors = scaler.fit_transform(features_vectors)
    class_labels = np.array(class_labels)
    
    return X_train, y_train, X_val, y_val, X_test, y_test
