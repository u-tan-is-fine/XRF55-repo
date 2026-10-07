import os
import shutil

import torch
import torch.utils.data as Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import csv
import time


def split_train_test_new(root_ns="./dataset/Raw_dataset/", dst_wr="./dataset/XRF_dataset/", split=14):

    dst_train_wifi = dst_wr + "train_data/WiFi/"

    dst_test_wifi = dst_wr + "test_data/WiFi/"

    if not os.path.exists(dst_train_wifi):
        os.makedirs(dst_train_wifi)


    if not os.path.exists(dst_test_wifi):
        os.makedirs(dst_test_wifi)



    for file in tqdm(os.listdir(root_ns + 'WiFi/')):
        filename = file.split(".")[0]  # act name
        fileidx = filename.split("_")[0]  # act idx
        actidx = int(filename.split("_")[2])
        if actidx <= split: # out of 20 samples of each action for each person, the first "zoo" are selected as the training set and the rest as the test set.
            os.symlink(os.path.abspath(root_ns + 'WiFi/' + filename + ".npy"), dst_train_wifi + filename + ".npy")
        else:
            os.symlink(os.path.abspath(root_ns + 'WiFi/' + filename + ".npy"), dst_test_wifi + filename + ".npy")

if __name__ == '__main__':

    split_train_test_new(root_ns="./dataset/Raw_dataset/", dst_wr="./dataset/XRF_dataset/", split=14)
