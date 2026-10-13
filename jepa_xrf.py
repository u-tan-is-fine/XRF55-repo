"""JEPA / JEPA-Anything (OPF) pretraining and evaluation on XRF55 Wi-Fi CSI.

Modes
  pretrain   : self-supervised JEPA pretraining on the training split (labels unused).
               --opf adds Orthogonal Predictive Factorization (JEPA-Anything, arXiv:2609.20800),
               using the official jepa-anything-core package for the OPF terms.
  probe      : linear probe on a frozen pretrained encoder (mean-pooled tokens).
  finetune   : fine-tune a pretrained encoder + linear head end-to-end (cross-entropy).
  supervised : same Transformer encoder + head trained from scratch (baseline without pretraining).

Data layout (made by split_train_test.py / generate_txt.py):
  <data_root>/train_data/WiFi/*.npy, <data_root>/test_data/WiFi/*.npy,
  <data_root>/<list_name>_train.txt, <data_root>/<list_name>_val.txt   (lines: name,subject,action)
Each sample: (270, 1000) = (3 receivers x 30 subcarriers x 3 antennas, 1000 time steps), amplitude.

Tokens: each receiver's 90 channels x `patch` time steps form one token -> 3 x (1000/patch) tokens.
The test split is used only once, at the end of probe / finetune / supervised. It is never used
for pretraining or for any model / epoch selection.
"""
import argparse
import copy
import json
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

N_RX, N_CH, T_LEN, N_CLS = 3, 90, 1000, 55
STATS_SEED = 0  # fixed so that every run uses the same normalization statistics


# ----------------------------------------------------------------------------- data
def read_list(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            name, subj, act = line.split(",")[:3]
            items.append((name, int(subj), int(act), int(name.split("_")[2])))
    return sorted(items)


def first_trials(items, n):
    """Few-shot labels: keep the first n trials (by trial index) of each (subject, action)."""
    if n <= 0:
        return items
    groups = defaultdict(list)
    for it in items:
        groups[(it[1], it[2])].append(it)
    out = []
    for key in sorted(groups):
        out += sorted(groups[key], key=lambda t: t[3])[:n]
    return out


def random_subset(items, n, seed):
    if n <= 0 or n >= len(items):
        return items
    rng = random.Random(seed)
    return sorted(rng.sample(items, n))


def channel_stats(items, wifi_dir, n=500):
    """Per-channel mean/std from (at most) n training samples. Training split only (no test leakage)."""
    sub = random_subset(items, n, STATS_SEED)
    s = np.zeros(N_RX * N_CH)
    s2 = np.zeros(N_RX * N_CH)
    cnt = 0
    for name, *_ in sub:
        x = np.load(os.path.join(wifi_dir, name + ".npy")).astype(np.float64)
        s += x.sum(1)
        s2 += (x ** 2).sum(1)
        cnt += x.shape[1]
    mean = s / cnt
    std = np.sqrt(np.maximum(s2 / cnt - mean ** 2, 1e-12))
    return mean.astype(np.float32), std.astype(np.float32)


class WifiSet(Dataset):
    def __init__(self, items, wifi_dir, mean=None, std=None):
        self.items, self.wifi_dir = items, wifi_dir
        self.mean = None if mean is None else mean[:, None]
        self.std = None if std is None else std[:, None]
        labels = [a - 1 for _, _, a, _ in items]
        if labels and not (min(labels) >= 0 and max(labels) < N_CLS):
            raise ValueError(f"labels out of range [{min(labels)}, {max(labels)}]")
        self.labels = labels

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        x = np.load(os.path.join(self.wifi_dir, self.items[i][0] + ".npy")).astype(np.float32)
        if x.shape != (N_RX * N_CH, T_LEN):
            raise ValueError(f"{self.items[i][0]}: shape {x.shape}")
        if self.mean is not None:
            x = (x - self.mean) / self.std
        return torch.from_numpy(x), self.labels[i]


# ----------------------------------------------------------------------------- model
def gather_tokens(h, idx):
    return torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]))


class Tokenizer(nn.Module):
    """(B, 270, 1000) -> (B, 3 * n_t, d). Token index = receiver * n_t + time_patch."""

    def __init__(self, d, patch):
        super().__init__()
        assert T_LEN % patch == 0
        self.patch, self.n_t, self.d = patch, T_LEN // patch, d
        self.ln = nn.LayerNorm(N_CH * patch)
        self.proj = nn.Linear(N_CH * patch, d)
        self.rx_emb = nn.Parameter(torch.randn(N_RX, d) * 0.02)
        self.t_emb = nn.Parameter(torch.randn(self.n_t, d) * 0.02)

    @property
    def n_tokens(self):
        return N_RX * self.n_t

    def pos(self):
        return (self.rx_emb[:, None, :] + self.t_emb[None, :, :]).reshape(-1, self.d)

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, N_RX, N_CH, self.n_t, self.patch).permute(0, 1, 3, 2, 4)
        x = x.reshape(b, self.n_tokens, N_CH * self.patch)
        return self.proj(self.ln(x)) + self.pos()


def transformer(d, depth, heads):
    layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=0.0,
                                       activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)


class Encoder(nn.Module):
    def __init__(self, d=128, depth=4, heads=4, patch=20):
        super().__init__()
        self.tok = Tokenizer(d, patch)
        self.blocks = transformer(d, depth, heads)
        self.norm = nn.LayerNorm(d)

    def forward(self, x, keep_idx=None):
        h = self.tok(x)
        if keep_idx is not None:
            h = gather_tokens(h, keep_idx)
        return self.norm(self.blocks(h))


class Predictor(nn.Module):
    """Context tokens + mask tokens (with target positions) -> predicted target representations.
    factors=(K, r): K branch heads on a shared trunk, output (B, Tt, K, r) (OPF);
    otherwise one head, output (B, Tt, d)."""

    def __init__(self, d, pdim, depth, heads, n_tokens, factors=None):
        super().__init__()
        self.inp = nn.Linear(d, pdim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pdim))
        self.pos = nn.Parameter(torch.randn(n_tokens, pdim) * 0.02)
        self.blocks = transformer(pdim, depth, heads)
        self.norm = nn.LayerNorm(pdim)
        self.factors = factors
        if factors is None:
            self.out = nn.Linear(pdim, d)
        else:
            k, r = factors
            self.heads = nn.ModuleList([nn.Linear(pdim, r) for _ in range(k)])

    def forward(self, ctx, ctx_idx, tgt_idx):
        b, tc, _ = ctx.shape
        c = self.inp(ctx) + self.pos[ctx_idx]
        m = self.mask_token.expand(b, tgt_idx.shape[1], -1) + self.pos[tgt_idx]
        h = self.norm(self.blocks(torch.cat([c, m], 1)))[:, tc:]
        if self.factors is None:
            return self.out(h)
        return torch.stack([head(h) for head in self.heads], dim=-2)


class Classifier(nn.Module):
    def __init__(self, enc, d):
        super().__init__()
        self.enc, self.head = enc, nn.Linear(d, N_CLS)

    def forward(self, x):
        return self.head(self.enc(x).mean(1))


def sample_masks(b, n_t, mode, span, device):
    """Return (ctx_idx, tgt_idx), each sorted by token position.
    time: one contiguous block of `span` time patches on all receivers is the target.
    rx  : one whole receiver is the target (simulates a missing receiver).
    mixed: time or rx, chosen per batch."""
    if mode == "mixed":
        mode = random.choice(["time", "rx"])
    n = N_RX * n_t
    tgt = torch.zeros(b, n, dtype=torch.int8)
    if mode == "time":
        start = torch.randint(0, n_t - span + 1, (b,))
        for i in range(b):
            for r in range(N_RX):
                tgt[i, r * n_t + start[i]: r * n_t + start[i] + span] = 1
    elif mode == "rx":
        rx = torch.randint(0, N_RX, (b,))
        for i in range(b):
            tgt[i, rx[i] * n_t:(rx[i] + 1) * n_t] = 1
    else:
        raise ValueError(mode)
    n_tgt = int(tgt[0].sum())
    order = torch.sort(tgt, dim=1, stable=True).indices  # context first, then target
    return order[:, : n - n_tgt].to(device), order[:, n - n_tgt:].to(device)


def var_hinge(z, min_std, eps=1e-6):
    """Mean over dims of relu(min_std - std), std over all samples/tokens (VICReg-style variance term)."""
    z = z.reshape(-1, z.shape[-1]).float()
    return F.relu(min_std - torch.sqrt(z.var(0, unbiased=False) + eps)).mean()


# ----------------------------------------------------------------------------- utils
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * p))


def loader(ds, args, shuffle, gen=None):
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers,
                      pin_memory=torch.cuda.is_available(), drop_last=False, generator=gen,
                      persistent_workers=args.num_workers > 0)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)


def env_info():
    return {"torch": torch.__version__, "numpy": np.__version__,
            "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    correct = n = 0
    for x, y in dl:
        pred = model(x.to(device, non_blocking=True)).argmax(1).cpu()
        correct += (pred == y).sum().item()
        n += y.numel()
    return correct / n


# ----------------------------------------------------------------------------- modes
def pretrain(args, data, device):
    train_items, wifi_tr, mean, std = data
    ds = WifiSet(train_items, wifi_tr, mean, std)
    g = torch.Generator().manual_seed(args.seed)
    dl = loader(ds, args, True, g)

    enc = Encoder(args.d, args.depth, args.heads, args.patch).to(device)
    tgt_enc = copy.deepcopy(enc).requires_grad_(False)
    n_t = enc.tok.n_t
    factors = None
    if args.opf:
        import jepa_anything_core as J  # pip install "git+https://github.com/Gen-Verse/JEPA-Anything.git#subdirectory=jepa-anything-core"
        assert args.d % args.K == 0
        factors = (args.K, args.d // args.K)
        opf = J.OrthogonalFactorProjection(state_dim=args.d, num_factors=args.K,
                                           factor_dim=args.d // args.K, learnable=True).to(device)
    pred = Predictor(args.d, args.pred_dim, args.pred_depth, args.heads, enc.tok.n_tokens, factors).to(device)
    params = list(enc.parameters()) + list(pred.parameters()) + (list(opf.parameters()) if args.opf else [])
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    total = args.epochs * len(dl)
    warm = args.warmup * len(dl)
    print(f"pretrain: {len(ds)} unlabeled samples, {enc.tok.n_tokens} tokens/sample, "
          f"encoder params {sum(p.numel() for p in enc.parameters())/1e6:.2f}M, opf={args.opf}")

    history, step, t0 = [], 0, time.time()
    for epoch in range(args.epochs):
        enc.train(); pred.train()
        acc = defaultdict(float)
        nb = 0
        for x, _ in dl:
            x = x.to(device, non_blocking=True)
            for gparam in opt.param_groups:
                gparam["lr"] = lr_at(step, total, warm, args.lr)
            ctx_idx, tgt_idx = sample_masks(x.shape[0], n_t, args.mask, args.span, device)
            with torch.no_grad():
                zt = gather_tokens(tgt_enc(x), tgt_idx)
                zt = F.layer_norm(zt, (zt.shape[-1],))
            zc = enc(x, ctx_idx)
            p = pred(zc, ctx_idx, tgt_idx)
            parts = {}
            if not args.opf:
                loss = F.mse_loss(p, zt)
                parts["pred"] = loss.item()
                if args.enc_var > 0:  # ablation: plain JEPA + encoder-variance term only
                    lv = var_hinge(zc, args.enc_min_std)
                    loss = loss + args.enc_var * lv
                    parts["enc_var"] = lv.item()
            else:
                tf = opf(zt)                              # (B, Tt, K, r), zt is already stop-gradient
                l_base = F.mse_loss(opf.compose(p), zt)   # monolithic term on the synthesized state
                L = J.jepa_anything_objective(p, tf, opf.analysis_basis(), zc,
                                              orthogonality_weight=args.lambda_orth,
                                              factor_activity_weight=args.lambda_fac,
                                              encoder_variance_weight=args.lambda_enc,
                                              factor_min_std=args.fac_min_std,
                                              encoder_min_std=args.enc_min_std)
                loss = l_base + L.total
                parts.update(base=l_base.item(), pred=L.prediction.item(), orth=L.orthogonality.item(),
                             fac=L.factor_activity.item(), enc_var=L.encoder_variance.item())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            m = args.ema + (1.0 - args.ema) * step / max(1, total)  # momentum: ema -> 1.0
            with torch.no_grad():
                for pt, po in zip(tgt_enc.parameters(), enc.parameters()):
                    pt.mul_(m).add_(po.detach(), alpha=1 - m)
            step += 1
            nb += 1
            acc["loss"] += loss.item()
            for k, v in parts.items():
                acc[k] += v
            acc["std_target"] += zt.reshape(-1, zt.shape[-1]).std(0).mean().item()
            acc["std_context"] += zc.reshape(-1, zc.shape[-1]).std(0).mean().item()
        rec = {"epoch": epoch, **{k: v / nb for k, v in acc.items()}, "time_s": round(time.time() - t0)}
        history.append(rec)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            torch.save({"online": enc.state_dict(), "target": tgt_enc.state_dict(),
                        "enc_args": {"d": args.d, "depth": args.depth, "heads": args.heads, "patch": args.patch},
                        "mean": mean, "std": std, "args": vars(args)},
                       os.path.join(args.out_dir, "encoder.pth"))
            save_json(history, os.path.join(args.out_dir, "history.json"))
    if args.opf:
        rep = J.audit_opf_geometry(opf, zt.reshape(-1, zt.shape[-1]).detach())
        save_json(rep.to_dict(), os.path.join(args.out_dir, "opf_audit.json"))
    save_json({"mode": "pretrain", "args": vars(args), "env": env_info(), "n_train": len(ds),
               "final": history[-1]}, os.path.join(args.out_dir, "result.json"))


def load_encoder(path, which, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    enc = Encoder(**ck["enc_args"])
    enc.load_state_dict(ck[which])
    return enc.to(device), ck["mean"], ck["std"], ck["enc_args"]["d"]


def probe(args, data, test, device):
    train_items, wifi_tr, _, _ = data
    test_items, wifi_te = test
    enc, mean, std, d = load_encoder(args.ckpt, args.which, device)
    enc.eval().requires_grad_(False)

    @torch.no_grad()
    def feats(items, wdir):
        X, Y = [], []
        for x, y in loader(WifiSet(items, wdir, mean, std), args, False):
            X.append(enc(x.to(device)).mean(1).cpu())
            Y.append(y)
        return torch.cat(X), torch.cat(Y)

    lab = first_trials(train_items, args.label_trials)
    Xtr, Ytr = feats(lab, wifi_tr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6           # feature standardization from the labeled train set only
    Xtr = ((Xtr - mu) / sd).to(device)
    Ytr = Ytr.to(device)
    head = nn.Linear(d, N_CLS).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=args.probe_wd)
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    for ep in range(args.probe_epochs):
        perm = torch.randperm(len(Xtr), generator=g).to(device)
        for i in range(0, len(Xtr), 256):
            idx = perm[i:i + 256]
            loss = F.cross_entropy(head(Xtr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        train_acc = (head(Xtr).argmax(1) == Ytr).float().mean().item()
    Xte, Yte = feats(test_items, wifi_te)
    with torch.no_grad():
        test_acc = (head(((Xte - mu) / sd).to(device)).argmax(1).cpu() == Yte).float().mean().item()
    res = {"mode": "probe", "encoder": args.which, "test_acc": test_acc, "train_acc": train_acc,
           "n_labeled": len(lab), "label_trials": args.label_trials, "n_test": len(test_items),
           "ckpt": args.ckpt, "args": vars(args), "env": env_info()}
    save_json(res, os.path.join(args.out_dir, f"probe_{args.which}_lt{args.label_trials}.json"))
    print(f"PROBE ({args.which}) labeled={len(lab)} train_acc={train_acc:.4f} TEST acc={test_acc:.4f} (n={len(test_items)})")


def train_classifier(args, data, test, device, pretrained):
    train_items, wifi_tr, mean, std = data
    test_items, wifi_te = test
    if pretrained:
        enc, mean, std, d = load_encoder(args.ckpt, args.which, device)
    else:
        enc, d = Encoder(args.d, args.depth, args.heads, args.patch).to(device), args.d
    model = Classifier(enc, d).to(device)
    lab = first_trials(train_items, args.label_trials)
    g = torch.Generator().manual_seed(args.seed)
    dl = loader(WifiSet(lab, wifi_tr, mean, std), args, True, g)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total, warm = args.epochs * len(dl), args.warmup * len(dl)
    mode = "finetune" if pretrained else "supervised"
    print(f"{mode}: labeled {len(lab)} samples, params {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    history, step, t0 = [], 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        ls = cor = n = 0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            for gparam in opt.param_groups:
                gparam["lr"] = lr_at(step, total, warm, args.lr)
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            ls += loss.item() * y.numel(); cor += (out.argmax(1) == y).sum().item(); n += y.numel()
        rec = {"epoch": epoch, "train_loss": ls / n, "train_acc": cor / n, "time_s": round(time.time() - t0)}
        history.append(rec)
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()), flush=True)
    test_acc = evaluate(model, loader(WifiSet(test_items, wifi_te, mean, std), args, False), device)
    torch.save(model.state_dict(), os.path.join(args.out_dir, f"{mode}_final.pth"))
    save_json(history, os.path.join(args.out_dir, f"{mode}_history.json"))
    res = {"mode": mode, "test_acc": test_acc, "n_labeled": len(lab), "label_trials": args.label_trials,
           "n_test": len(test_items), "epochs": args.epochs, "args": vars(args), "env": env_info()}
    save_json(res, os.path.join(args.out_dir, f"{mode}_lt{args.label_trials}.json"))
    print(f"{mode.upper()} labeled={len(lab)} TEST acc={test_acc:.4f} (n={len(test_items)}, last epoch)")


# ----------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description="JEPA / JEPA-Anything on XRF55 Wi-Fi")
    p.add_argument("--mode", choices=["pretrain", "probe", "finetune", "supervised"], required=True)
    p.add_argument("--data_root", default="./dataset/XRF_dataset/")
    p.add_argument("--list_name", default="dml")
    p.add_argument("--out_dir", default="./result/jepa/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=5, help="warm-up epochs")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--norm", choices=["channel", "none"], default="channel",
                   help="channel: per-channel standardization with training-split statistics")
    p.add_argument("--limit", type=int, default=0, help="debug: random subset of N train and N test samples")
    p.add_argument("--label_trials", type=int, default=0,
                   help="few-shot: labeled trials per (subject, action) for probe/finetune/supervised (0 = all 14)")
    # encoder
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--patch", type=int, default=20, help="time steps per token (1000 must be divisible)")
    # pretraining
    p.add_argument("--pred_dim", type=int, default=128)
    p.add_argument("--pred_depth", type=int, default=2)
    p.add_argument("--mask", choices=["time", "rx", "mixed"], default="mixed")
    p.add_argument("--span", type=int, default=15, help="time patches per target block (mask=time)")
    p.add_argument("--ema", type=float, default=0.996, help="initial EMA momentum, increased linearly to 1.0")
    p.add_argument("--enc_var", type=float, default=0.0, help="ablation: weight of encoder-variance term without OPF")
    p.add_argument("--opf", action="store_true", help="JEPA-Anything: orthogonal predictive factorization")
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--lambda_orth", type=float, default=0.10)
    p.add_argument("--lambda_fac", type=float, default=0.05)
    p.add_argument("--lambda_enc", type=float, default=0.02)
    p.add_argument("--fac_min_std", type=float, default=0.1)
    p.add_argument("--enc_min_std", type=float, default=0.1)
    # evaluation
    p.add_argument("--ckpt", default=None, help="encoder.pth from pretrain (probe / finetune)")
    p.add_argument("--which", choices=["target", "online"], default="target", help="which pretrained encoder to use")
    p.add_argument("--probe_epochs", type=int, default=200)
    p.add_argument("--probe_wd", type=float, default=1e-4)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(env_info(), vars(args))

    train_items = read_list(os.path.join(args.data_root, f"{args.list_name}_train.txt"))
    test_items = read_list(os.path.join(args.data_root, f"{args.list_name}_val.txt"))
    if args.limit:
        train_items = random_subset(train_items, args.limit, STATS_SEED)
        test_items = random_subset(test_items, args.limit, STATS_SEED)
    wifi_tr = os.path.join(args.data_root, "train_data", "WiFi")
    wifi_te = os.path.join(args.data_root, "test_data", "WiFi")
    print(f"train {len(train_items)}, test {len(test_items)}, subjects {sorted({s for _, s, _, _ in train_items})}")

    mean = std = None
    if args.mode in ("pretrain", "supervised") and args.norm == "channel":
        mean, std = channel_stats(train_items, wifi_tr)
    data = (train_items, wifi_tr, mean, std)
    test = (test_items, wifi_te)

    if args.mode == "pretrain":
        pretrain(args, data, device)
    elif args.mode == "probe":
        probe(args, data, test, device)
    elif args.mode == "finetune":
        train_classifier(args, data, test, device, pretrained=True)
    else:
        train_classifier(args, data, test, device, pretrained=False)


if __name__ == "__main__":
    main()
