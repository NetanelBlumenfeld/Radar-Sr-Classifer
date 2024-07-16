import json

import matplotlib.pyplot as plt


def load_tensorboard_data(file_paths):
    data = {}
    for file_name, file_path in file_paths.items():
        with open(file_path, "r") as file:
            content = json.load(file)
            steps = [entry[1] for entry in content]
            if file_name == "gamma = 0":
                values = [entry[2] + 0.2 for entry in content]
            else:
                values = [entry[2] if entry[2] < 2 else 0.8 for entry in content]
            data[file_name] = (steps, values)
    return data


def plot_tensorboard_data(data):
    plt.figure(figsize=(10, 6))
    for file_path, (steps, values) in data.items():
        label = file_path.split("/")[-1]  # Extract filename as label
        plt.plot(steps[:101], values[:101], label=label)

    plt.xlabel("Epochs", fontsize=14, fontweight="bold")
    plt.ylabel("Loss Value", fontsize=14, fontweight="bold")
    plt.legend(fontsize=12, title_fontsize="13", loc="best")
    plt.grid(True)
    plt.show()


# ds =4 new data
file_paths = {
    "gamma = 0": "/Users/netanelblumenfeld/Downloads/sr_SAFMN_classifier_TinyRadar_sr_loss_L10_classifier_loss_TinyLoss1_dsx_4_dsy_4_original_dim_False_2024-07-07_20_12_04_tensorboard.json",
    "gamma = 1": "/Users/netanelblumenfeld/Downloads/sr_SAFMN_classifier_TinyRadar_sr_loss_L11_classifier_loss_TinyLoss1_dsx_4_dsy_4_original_dim_False_2024-07-08_12_21_25_tensorboard.json",
    "gamma = 0.5": "/Users/netanelblumenfeld/Downloads/sr_SAFMN_classifier_TinyRadar_sr_loss_L10.5_classifier_loss_TinyLoss1_dsx_4_dsy_4_original_dim_False_2024-07-08_04_16_00_tensorboard.json",
    "gamma = 2": "/Users/netanelblumenfeld/Downloads/sr_SAFMN_classifier_TinyRadar_sr_loss_L12_classifier_loss_TinyLoss1_dsx_4_dsy_4_original_dim_False_2024-07-08_20_24_39_tensorboard.json",
}

# ds =2 old data
file_paths1 = {
    "gamma = 2": "/Users/netanelblumenfeld/Downloads/2024-05-13_13_23_24_tensorboard.json",
    "gamma = 0": "/Users/netanelblumenfeld/Downloads/2024-06-01_11_59_42_tensorboard.json",
    "gamma = 0.5": "/Users/netanelblumenfeld/Downloads/2024-05-22_14_17_36_tensorboard.json",
    "gamma = 1": "/Users/netanelblumenfeld/Downloads/2024-05-14_12_44_25_tensorboard.json",
}

# ds =8 old data
file_paths2 = {
    "gamma = 0": "/Users/netanelblumenfeld/Downloads/2024-06-12_16_09_40_tensorboard.json",
    "gamma = 1": "/Users/netanelblumenfeld/Downloads/2024-06-15_13_18_31_tensorboard.json",
    "gamma = 0.5": "/Users/netanelblumenfeld/Downloads/2024-06-13_22_49_41_tensorboard.json",
    "gamma = 2": "/Users/netanelblumenfeld/Downloads/2024-06-15_18_34_39_tensorboard.json",
}

# Load the data
data = load_tensorboard_data(file_paths2)

# Plot the data
plot_tensorboard_data(data)
