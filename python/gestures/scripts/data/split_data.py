import os

import numpy as np
from listutils import ensure_dir


def split_data(data_feat_path, data_feat_split_path, people, gestures):
    for idx in people:
        person = "p" + str(idx)
        print("Person:", person)
        if "_" in person:
            person_path_save = person.split("_")[0] + person.split("_")[1]
        else:
            person_path_save = person

        person_path = os.path.join(data_feat_path, str(person))
        for gesture_file in os.listdir(person_path):
            if not gesture_file.endswith("npy"):
                continue
            gesture_path = os.path.join(person_path, gesture_file)
            data = np.load(gesture_path)
            for i in range(data.shape[0]):
                sample = data[i]
                gesture_name = gesture_file.split("_")[0]
                file_name = (
                    person_path_save + "_" + gesture_name + "_" + str(i) + ".npy"
                )
                path_to_save = os.path.join(data_feat_split_path, file_name)
                np.save(path_to_save, sample)


if __name__ == "__main__":
    data_feat_path = "/Users/netanelblumenfeld/Downloads/11G/data_feat"
    data_feat_split_path = "/Users/netanelblumenfeld/Downloads/11G/data_feat_split"
    ensure_dir(data_feat_split_path)

    people = list(range(1, 26, 1))
    singleuserlist = list(map(lambda x: "0_" + str(x), list(range(1, 21, 1))))
    people += singleuserlist
    gestures = [
        "PinchIndex",
        "PinchPinky",
        "FingerSlider",
        "FingerRub",
        "SlowSwipeRL",
        "FastSwipeRL",
        "Push",
        "Pull",
        "PalmTilt",
        "Circle",
        "PalmHold",
        "NoHand",
        "RandomGesture",
    ]
    split_data(data_feat_path, data_feat_split_path, people, gestures)
