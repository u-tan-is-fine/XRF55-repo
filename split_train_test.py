"""Wi-Fi only train/test split for XRF55 (Scene 1).

Original: airslab2020/XRF55-repo split_train_test.py (MIT License).
Changes from the original:
  * Only the WiFi modality is handled (RFID and mmWave are not read or copied).
  * Files are linked (symlink) instead of copied by default, so no extra disk
    space is used (the Wi-Fi data is ~25 GB; /kaggle/working is limited).
  * Only the subjects given by --subjects are used (default 1-11).
  * The first `split` trials of each (subject, action) pair, ordered by trial
    index, form the training set and the remaining trials form the test set.
    This does not depend on whether trial indices start at 0 or 1.
  * Fixed the original bugs (undefined `zoo`, `os.mkdirs`).

File name format: <subject>_<action>_<trial>.npy
"""
import argparse
import os
import shutil
from collections import defaultdict


def parse_subjects(text):
    """'1-11' -> {1,...,11};  '1,3,5-7' -> {1,3,5,6,7}."""
    out = set()
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def place(src, dst, copy):
    """Create dst from src as a symlink (default) or a copy."""
    if os.path.lexists(dst):
        return
    if copy:
        shutil.copy(src, dst)
        return
    try:
        os.symlink(os.path.abspath(src), dst)
    except OSError:  # e.g. symlinks not permitted -> fall back to copy
        shutil.copy(src, dst)


def split_train_test_wifi(root_ns="./dataset/Raw_dataset/",
                          dst_wr="./dataset/XRF_dataset/",
                          split=14, subjects=range(1, 12), copy=False):
    src_dir = os.path.join(root_ns, "WiFi")
    dst_train = os.path.join(dst_wr, "train_data", "WiFi")
    dst_test = os.path.join(dst_wr, "test_data", "WiFi")
    os.makedirs(dst_train, exist_ok=True)
    os.makedirs(dst_test, exist_ok=True)

    subjects = set(subjects)
    groups = defaultdict(list)  # (subject, action) -> [(trial, filename)]
    skipped = 0
    for file in os.listdir(src_dir):
        if not file.endswith(".npy"):
            continue
        name = file[:-4]
        parts = name.split("_")
        if len(parts) != 3:
            skipped += 1
            continue
        subj, act, trial = (int(p) for p in parts)
        if subj not in subjects:
            skipped += 1
            continue
        groups[(subj, act)].append((trial, name))

    n_train = n_test = 0
    bad = []
    for key in sorted(groups):
        items = sorted(groups[key])  # ordered by trial index
        if len(items) != 20:
            bad.append((key, len(items)))
        for rank, (trial, name) in enumerate(items):
            src = os.path.join(src_dir, name + ".npy")
            if rank < split:
                place(src, os.path.join(dst_train, name + ".npy"), copy)
                n_train += 1
            else:
                place(src, os.path.join(dst_test, name + ".npy"), copy)
                n_test += 1

    print(f"subjects used: {sorted({k[0] for k in groups})}")
    print(f"(subject, action) groups: {len(groups)}")
    print(f"train: {n_train}, test: {n_test}, skipped (other subjects/names): {skipped}")
    if bad:
        print(f"WARNING: {len(bad)} groups do not have 20 trials, e.g. {bad[:5]}")
    # Expected for Scene 1, 11 subjects x 55 actions x 20 trials: 8470 / 3630
    if len(subjects) == 11:
        if (n_train, n_test) != (8470, 3630):
            print("WARNING: expected train=8470, test=3630 for 11 subjects")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./dataset/Raw_dataset/",
                   help="directory that contains the WiFi/ folder")
    p.add_argument("--dst", default="./dataset/XRF_dataset/")
    p.add_argument("--split", type=int, default=14,
                   help="number of training trials per (subject, action)")
    p.add_argument("--subjects", default="1-11")
    p.add_argument("--copy", action="store_true",
                   help="copy files instead of creating symlinks")
    a = p.parse_args()
    split_train_test_wifi(a.root, a.dst, a.split, parse_subjects(a.subjects), a.copy)
