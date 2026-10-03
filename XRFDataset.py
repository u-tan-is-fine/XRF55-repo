import logging
import os
from collections import defaultdict
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
log = logging.getLogger(__name__)

class XRFBertDatasetNewMix(Dataset):
    def __init__(self, file_path='./dataset/XRFDataset/', is_train=True, scene='dml'):
        super(XRFBertDatasetNewMix, self).__init__()
        self.word_list = np.load("./word2vec/bert_new_sentence_large_uncased.npy")
        self.file_path = file_path
        self.is_train = is_train
        self.scene = scene
        if self.is_train:
            self.file = self.file_path + self.scene + '_train.txt'
        else:
            self.file = self.file_path + self.scene + '_val.txt'
        file = open(self.file)
        val_list = file.readlines()
        self.data = {
            'file_name': list(),
            'label': list()
        }
        self.path = self.file_path + self.scene + '_new_data/'
        for string in val_list:
            self.data['file_name'].append(string.split(',')[0])
            self.data['label'].append(int(string.split(',')[2]) - 1)
        log.info("load XRF dataset")

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        file_name = self.data['file_name'][idx]
        label = self.data['label'][idx]
        vector = self.word_list[label]

        wifi_data = load_wifi(file_name, self.is_train, path=self.path)
        rfid_data = load_rfid(file_name, self.is_train, path=self.path)
        mmwave_data = load_mmwave(file_name, self.is_train, path=self.path)
        return wifi_data, rfid_data, mmwave_data, label, vector



def load_rfid(filename, is_train, path='./dataset/XRFDataset/'):
    if is_train:
        path = path + 'train_data/'
    else:
        path = path + 'test_data/'
    record = np.load(path + 'RFID/' + filename + ".npy")
    return torch.from_numpy(record).float()


def load_wifi(filename, is_train, path='./dataset/XRFDataset/'):
    if is_train:
        path = path + 'train_data/'
    else:
        path = path + 'test_data/'
    record = np.load(path + 'WiFi/' + filename + ".npy")
    return torch.from_numpy(record).float()


def load_mmwave(filename, is_train, path='./dataset/XRFDataset/'):
    if is_train:
        path = path + 'train_data/'
    else:
        path = path + 'test_data/'
    mmWave_data = np.load(path + 'mmWave/' + filename + ".npy")
    return torch.from_numpy(mmWave_data).float()


# ---------------------------------------------------------------------------
# Wi-Fi only Dataset (added). Used by dml_train.py. The classes above are the
# original 3-modality code and are left unchanged (dml_eval.py still uses them).
#
# List file format (written by generate_txt.py):  <name>,<subject>,<action>
# Data file: <wifi_dir>/<name>.npy, shape (270, 1000), amplitude only.
# The label is action - 1 (actions are numbered 1..55).
# ---------------------------------------------------------------------------
NUM_CLASSES = 55
INPUT_SHAPE = (270, 1000)


def read_list(list_file):
    """Return [(name, subject, action)] from a list file."""
    items = []
    with open(list_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, subj, act = line.split(",")
            items.append((name, int(subj), int(act)))
    return items


def holdout_last_trials(items, k):
    """Split items into (rest, held_out): the last k trials (by trial index)
    of every (subject, action) pair are held out. k = 0 returns (items, [])."""
    if k <= 0:
        return list(items), []
    groups = defaultdict(list)
    for it in items:
        trial = int(it[0].split("_")[2])
        groups[(it[1], it[2])].append((trial, it))
    rest, held = [], []
    for key in sorted(groups):
        ordered = [it for _, it in sorted(groups[key])]
        if len(ordered) <= k:
            raise ValueError(f"{key}: only {len(ordered)} trials, cannot hold out {k}")
        rest += ordered[:-k]
        held += ordered[-k:]
    return rest, held


class XRFWifiDataset(Dataset):
    def __init__(self, items, wifi_dir):
        self.items = list(items)
        self.wifi_dir = wifi_dir
        labels = [a - 1 for _, _, a in self.items]
        if labels and not (0 <= min(labels) and max(labels) < NUM_CLASSES):
            raise ValueError(f"label range [{min(labels)}, {max(labels)}] is outside "
                             f"[0, {NUM_CLASSES - 1}]; check the action numbering")
        self.labels = labels

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        name = self.items[idx][0]
        x = np.load(os.path.join(self.wifi_dir, name + ".npy"))
        if x.shape != INPUT_SHAPE:
            raise ValueError(f"{name}: shape {x.shape}, expected {INPUT_SHAPE}")
        return torch.from_numpy(x).float(), self.labels[idx]
