from pathlib import Path
import json
import random
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "artifacts"
EMB_PATH = OUT_DIR / "train_embeddings.pt"
LABEL_PATH = OUT_DIR / "train_labels.pt"
META_PATH = OUT_DIR / "meta.json"
BEST_PATH = OUT_DIR / "best_head.pt"

BATCH_SIZE = 2048
EPOCHS = 25
LR = 3e-3
WEIGHT_DECAY = 1e-2
VAL_RATIO = 0.1
PATIENCE = 7
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()

def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

class EmbeddingDataset(Dataset):
    def __init__(self, embs, labels):
        self.embs = embs
        self.labels = labels

    def __len__(self):
        return self.embs.shape[0]

    def __getitem__(self, idx):
        return self.embs[idx], self.labels[idx]

@torch.no_grad()
def evaluate(head, loader, criterion):
    head.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True).float()
        y = y.to(DEVICE, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=USE_AMP):
            logits = head(x)
            loss = criterion(logits, y)
        bs = y.size(0)
        total_loss += loss.item() * bs
        total_acc += (logits.argmax(dim=-1) == y).float().mean().item() * bs
        n += bs

    return total_loss / max(1, n), total_acc / max(1, n)

def main():
    embs = torch.load(EMB_PATH, map_location="cpu")
    labels = torch.load(LABEL_PATH, map_location="cpu")

    if embs.dtype != torch.float32:
        embs = embs.float()

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    emb_dim = int(meta["emb_dim"])
    num_classes = int(meta["num_speakers"])

    ds = EmbeddingDataset(embs, labels)
    val_size = max(1, int(len(ds) * VAL_RATIO))
    train_size = len(ds) - val_size

    g = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=g)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    head = nn.Sequential(
        nn.LayerNorm(emb_dim),
        nn.Linear(emb_dim, num_classes)
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_val = float("inf")
    bad_epochs = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        head.train()
        total_loss = 0.0
        total_acc = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS}")
        for x, y in pbar:
            x = x.to(DEVICE, non_blocking=True).float()
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=USE_AMP):
                logits = head(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_acc += (logits.argmax(dim=-1) == y).float().mean().item() * bs
            n += bs
            pbar.set_postfix(loss=loss.item())

        train_loss = total_loss / max(1, n)
        train_acc = total_acc / max(1, n)

        val_loss, val_acc = evaluate(head, val_loader, criterion)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save({
                "head_state_dict": head.state_dict(),
                "emb_dim": emb_dim,
                "num_classes": num_classes,
                "meta": meta
            }, BEST_PATH)
            print(f"saved: {BEST_PATH}")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("early stopping")
                break

    with open(OUT_DIR / "head_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()