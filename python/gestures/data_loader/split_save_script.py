import os

import numpy as np
from gestures import config as cfg
from gestures.data_loader.tiny_data import load_tiny_data
from gestures.setup import get_pc_cgf
from sklearn.model_selection import train_test_split


def save_data_splits(base_dir: str, X: np.ndarray, y: np.ndarray):
    # Split data into training and test set (80% train, 20% test)
    print("starting...")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Split the temp data into validation and test sets (50% val, 50% test)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Create directories for the splits if they don't exist
    train_dir = os.path.join(base_dir, "data_feat/train")
    val_dir = os.path.join(base_dir, "data_feat/val")
    test_dir = os.path.join(base_dir, "data_feat/test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Save the splits to their respective directories in .npy format
    np.save(os.path.join(train_dir, "X.npy"), X_train)
    np.save(os.path.join(train_dir, "y.npy"), y_train)

    np.save(os.path.join(val_dir, "X.npy"), X_val)
    np.save(os.path.join(val_dir, "y.npy"), y_val)

    np.save(os.path.join(test_dir, "X.npy"), X_test)
    np.save(os.path.join(test_dir, "y.npy"), y_test)

    print("Data has been split and saved in .npy format to:", base_dir)


if __name__ == "__main__":
    pc = cfg.pc
    data_dir, output_dir, device = get_pc_cgf(pc)
    base_dir = data_dir.split("/11G/")[0]

    dp_cfg = cfg.data_preprocessing_cfg
    data_cfg = cfg.data_cfg

    X, y = load_tiny_data(
        data_dir, data_cfg["people"], data_cfg["gestures"], data_cfg["data_type"]
    )
    save_data_splits(base_dir, X, y)
