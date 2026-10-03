"""Wi-Fi only training for XRF55 (cross-entropy only, no DML, no BERT loss).

Original: airslab2020/XRF55-repo dml_train.py (MIT License).
Changes from the original:
  * Only the Wi-Fi model is trained; RFID/mmWave models and data are not used.
  * Loss is cross-entropy only (no KL / mutual-learning term, no BERT L1 term).
  * Removed the missing `import mmd_loss`.
  * Command-line options for paths, num_workers (default 4), seed, etc.
  * Per-epoch checkpoint and --resume (Kaggle sessions are limited to 12 h).
  * Optional validation: --val_trials K holds out the last K of the 14 training
    trials of every (subject, action) pair; the test set is NOT used for model
    selection. The test set is evaluated once, after the last epoch.
  * Training loss is averaged per sample (the original divided a sum of batch
    means by the number of samples, so the printed value was off by the batch size).
  * mkdir -> makedirs(exist_ok=True); torch.autograd.Variable removed;
    retain_graph removed; falls back to CPU when CUDA is unavailable.

Dataset: XRFWifiDataset in XRFDataset.py.
Input: ./dataset/XRF_dataset/{train_data,test_data}/WiFi/*.npy (made by
split_train_test.py) and dml_train.txt / dml_val.txt (made by generate_txt.py).
No normalization is applied (raw amplitude, as in the distributed code).
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import resnet1d  # noqa: E402
from XRFDataset import (NUM_CLASSES, XRFWifiDataset, holdout_last_trials,  # noqa: E402
                          read_list)


def parse_args():
    p = argparse.ArgumentParser(description="XRF55 Wi-Fi only training")
    p.add_argument("--data_root", default="./dataset/XRF_dataset/",
                   help="folder with train_data/WiFi and test_data/WiFi")
    p.add_argument("--list_dir", default=None,
                   help="folder with <list_name>_train.txt / _val.txt (default: data_root)")
    p.add_argument("--list_name", default="dml")
    p.add_argument("--out_dir", default="./result/params/train_wifi/")
    p.add_argument("--arch", choices=["mutual", "plain"], default="mutual",
                   help="mutual: resnet18_mutual (the Wi-Fi model used by the distributed "
                        "dml_train.py/dml_eval.py); plain: resnet18")
    p.add_argument("--epoch", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--lr_step", type=int, default=40, help="lr x0.5 every lr_step epochs")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_trials", type=int, default=0,
                   help="hold out the last K training trials per (subject, action) as validation")
    p.add_argument("--limit", type=int, default=0,
                   help="debug: use only the first N samples of each list (0 = all)")
    p.add_argument("--resume", action="store_true", help="resume from out_dir/ckpt.pth")
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def forward_logits(model, x):
    out = model(x)
    return out[0] if isinstance(out, tuple) else out  # *_mutual also returns a 1024-d vector


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    n, loss_sum, correct = 0, 0.0, 0
    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = forward_logits(model, x)
        loss_sum += criterion(logits, y).item() * y.size(0)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        n += y.size(0)
        for t, q in zip(y.tolist(), pred.tolist()):
            conf[t, q] += 1
    return loss_sum / n, correct / n, conf


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    list_dir = args.list_dir or args.data_root

    print(f"torch {torch.__version__}, numpy {np.__version__}, device {device}, seed {args.seed}")
    print(vars(args))

    train_items = read_list(os.path.join(list_dir, f"{args.list_name}_train.txt"))
    test_items = read_list(os.path.join(list_dir, f"{args.list_name}_val.txt"))  # test list
    if args.limit > 0:
        train_items, test_items = train_items[:args.limit], test_items[:args.limit]
    train_items, val_items = holdout_last_trials(train_items, args.val_trials)
    train_dir = os.path.join(args.data_root, "train_data", "WiFi")
    test_dir = os.path.join(args.data_root, "test_data", "WiFi")

    train_set = XRFWifiDataset(train_items, train_dir)
    val_set = XRFWifiDataset(val_items, train_dir) if val_items else None
    test_set = XRFWifiDataset(test_items, test_dir)
    print(f"train {len(train_set)}, val {len(val_set) if val_set else 0}, test {len(test_set)}; "
          f"subjects in train: {sorted({s for _, s, _ in train_items})}")

    g = torch.Generator()
    g.manual_seed(args.seed)
    kw = dict(num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
              persistent_workers=args.num_workers > 0, worker_init_fn=worker_init)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              drop_last=False, generator=g, **kw)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kw) if val_set else None
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kw)

    model = (resnet1d.resnet18_mutual() if args.arch == "mutual" else resnet1d.resnet18()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    history, start_epoch = [], 0
    ckpt_path = os.path.join(args.out_dir, "ckpt.pth")
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        history, start_epoch = ck["history"], ck["epoch"] + 1
        print(f"resumed from epoch {start_epoch} (shuffle order / RNG state is not restored)")

    t0 = time.time()
    for epoch in range(start_epoch, args.epoch):
        model.train()
        n, loss_sum, correct = 0, 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = forward_logits(model, x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)
        scheduler.step()
        rec = {"epoch": epoch, "train_loss": loss_sum / n, "train_acc": correct / n}
        if val_loader is not None:
            rec["val_loss"], rec["val_acc"], _ = evaluate(model, val_loader, device, criterion)
        history.append(rec)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()),
              f"time={time.time() - t0:.0f}s", flush=True)
        save_atomic({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "scheduler": scheduler.state_dict(), "history": history, "epoch": epoch,
                     "args": vars(args)}, ckpt_path)
        with open(os.path.join(args.out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

    # Test set: evaluated once, with the weights of the last epoch.
    torch.save(model.state_dict(), os.path.join(args.out_dir, "wifi_final.pth"))
    test_loss, test_acc, conf = evaluate(model, test_loader, device, criterion)
    np.save(os.path.join(args.out_dir, "test_conf_matrix.npy"), conf)
    result = {"test_acc": test_acc, "test_loss": test_loss, "n_train": len(train_set),
              "n_val": len(val_set) if val_set else 0, "n_test": len(test_set),
              "epochs": args.epoch, "args": vars(args), "torch": torch.__version__}
    with open(os.path.join(args.out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=1)
    print(f"TEST acc={test_acc:.4f} loss={test_loss:.4f} (n={len(test_set)})")


if __name__ == "__main__":
    main()
