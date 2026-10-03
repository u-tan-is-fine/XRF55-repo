"""Generate train/val list files from the Wi-Fi split.

Original: airslab2020/XRF55-repo generate_txt.py (MIT License).
Change from the original: the file lists are built from the WiFi folders
(the original read the RFID folders). Output format is unchanged:
    <filename without .npy>,<subject>,<action>
"""
import argparse
import os


def _list_wifi(src):
    lines = []
    for file in sorted(os.listdir(src)):
        if not file.endswith(".npy"):
            continue
        filename = file[:-4]
        subj, act, _ = filename.split("_")  # subject, action, trial
        lines.append(f"{filename},{subj},{act}\n")
    return lines


def generate_txt_mix(data_src_path, txt_save_path, txt_name):
    train_src = os.path.join(data_src_path, "train_data", "WiFi")
    test_src = os.path.join(data_src_path, "test_data", "WiFi")

    train_list = _list_wifi(train_src)
    val_list = _list_wifi(test_src)

    with open(os.path.join(txt_save_path, txt_name + "_train.txt"), "w") as f:
        f.writelines(train_list)
    print("train_dataset len: " + str(len(train_list)))
    with open(os.path.join(txt_save_path, txt_name + "_val.txt"), "w") as f:
        f.writelines(val_list)
    print("test_dataset len: " + str(len(val_list)))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="./dataset/XRF_dataset/")
    p.add_argument("--out", default="./dataset/XRF_dataset/")
    p.add_argument("--name", default="dml")
    a = p.parse_args()
    generate_txt_mix(a.src, a.out, a.name)
