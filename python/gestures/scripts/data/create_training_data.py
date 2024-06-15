import os
import random
import shutil


def split_data(
    source_folder,
    train_folder,
    val_folder,
    test_folder,
    train_split=0.7,
    val_split=0.2,
    test_split=0.1,
):
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    files = os.listdir(source_folder)
    random.shuffle(files)

    total_files = len(files)
    train_count = int(total_files * train_split)
    val_count = int(total_files * val_split)
    test_count = total_files - train_count - val_count

    train_files = files[:train_count]
    val_files = files[train_count : train_count + val_count]
    test_files = files[train_count + val_count :]

    for file in train_files:
        shutil.copy(os.path.join(source_folder, file), train_folder)

    for file in val_files:
        shutil.copy(os.path.join(source_folder, file), val_folder)

    for file in test_files:
        shutil.copy(os.path.join(source_folder, file), test_folder)


source_folder = "/Users/netanelblumenfeld/Downloads/11G/data_feat_split"
train_folder = "/Users/netanelblumenfeld/Downloads/11G/train"
val_folder = "/Users/netanelblumenfeld/Downloads/11G/val"
test_folder = "/Users/netanelblumenfeld/Downloads/11G/test"

split_data(source_folder, train_folder, val_folder, test_folder)
print("finish")
