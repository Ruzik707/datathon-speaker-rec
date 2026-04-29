from pathlib import Path
import json
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.dataio.dataio import read_audio

# ===================== CONFIG =====================
DATA_PATH = Path(__file__).parent.parent / "data" / "raw" / "train_part_1" / "train"
OUT_DIR = Path(__file__).parent.parent / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
BATCH_SIZE = 16
EPOCHS = 4
LR = 1e-3
WEIGHT_DECAY = 1e-2
VAL_RATIO = 0.1
PATIENCE = 2
NUM_WORKERS = 2
SEED = 42
NUM_FILES_TO_TRAIN = 500
TOP_K = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===================== SEED =====================
def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

# ===================== AUGMENTATION =====================
class SimpleAudioAugmentor:
    def __init__(self, noise_p=0.6, gain_p=0.5, shift_p=0.5,
                 noise_level=0.01, gain_min=0.8, gain_max=1.2, max_shift_ratio=0.1):
        self.noise_p = noise_p
        self.gain_p = gain_p
        self.shift_p = shift_p
        self.noise_level = noise_level
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.max_shift_ratio = max_shift_ratio

    def add_noise(self, wav):
        std = wav.std().clamp_min(1e-6)
        return wav + torch.randn_like(wav) * std * self.noise_level

    def apply_gain(self, wav):
        gain = random.uniform(self.gain_min, self.gain_max)
        return wav * gain

    def time_shift(self, wav):
        if wav.numel() < 2:
            return wav
        max_shift = max(1, int(wav.shape[-1] * self.max_shift_ratio))
        shift = random.randint(-max_shift, max_shift)
        return torch.roll(wav, shifts=shift, dims=-1)

    def __call__(self, wav):
        if random.random() < self.noise_p:
            wav = self.add_noise(wav)
        if random.random() < self.gain_p:
            wav = self.apply_gain(wav)
        if random.random() < self.shift_p:
            wav = self.time_shift(wav)
        return wav

augmentor = SimpleAudioAugmentor()

# ===================== DATASET =====================
class SpeakerDataset(Dataset):
    def __init__(self, file_paths, speaker_to_idx, train=True):
        self.file_paths = list(file_paths)
        self.speaker_to_idx = speaker_to_idx
        self.train = train

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        wav = read_audio(str(path))

        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        elif wav.dim() > 2:
            wav = wav.squeeze()

        wav = wav.float()

        if self.train:
            wav = augmentor(wav)

        label = self.speaker_to_idx[path.parent.name]
        return wav, label, str(path)

def collate_fn(batch):
    wavs, labels, paths = zip(*batch)
    lengths = torch.tensor([w.shape[-1] for w in wavs], dtype=torch.float32)
    max_len = int(lengths.max().item())

    padded = []
    for w in wavs:
        if w.shape[-1] < max_len:
            w = torch.nn.functional.pad(w, (0, max_len - w.shape[-1]))
        padded.append(w)

    wavs = torch.stack(padded, dim=0)
    wav_lens = lengths / max_len
    labels = torch.tensor(labels, dtype=torch.long)
    return wavs, wav_lens, labels, paths

# ===================== METRICS =====================
@torch.no_grad()
def batch_accuracy(logits, targets):
    return (logits.argmax(dim=-1) == targets).float().mean().item()

@torch.no_grad()
def evaluate(encoder, head, loader, criterion):
    encoder.eval()
    head.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_n = 0

    for wavs, wav_lens, labels, _ in tqdm(loader, desc="val", leave=False):
        wavs = wavs.to(DEVICE, non_blocking=True)
        wav_lens = wav_lens.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        emb = encoder.encode_batch(wavs, wav_lens=wav_lens)
        if emb.dim() == 3:
            emb = emb.squeeze(1)

        logits = head(emb)
        loss = criterion(logits, labels)

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc += batch_accuracy(logits, labels) * bs
        total_n += bs

    return total_loss / max(1, total_n), total_acc / max(1, total_n)

# ===================== MAIN =====================
def main():
    audio_files = sorted(DATA_PATH.glob("**/*.flac"))
    if NUM_FILES_TO_TRAIN is not None:
        audio_files = audio_files[:NUM_FILES_TO_TRAIN]

    speakers = sorted({p.parent.name for p in audio_files})
    speaker_to_idx = {spk: i for i, spk in enumerate(speakers)}
    num_speakers = len(speakers)

    print(f"Files: {len(audio_files)}, Speakers: {num_speakers}")

    idxs = list(range(len(audio_files)))
    random.shuffle(idxs)
    val_size = max(1, int(len(idxs) * VAL_RATIO))
    val_idxs = idxs[:val_size]
    train_idxs = idxs[val_size:]

    train_files = [audio_files[i] for i in train_idxs]
    val_files = [audio_files[i] for i in val_idxs]

    train_ds = SpeakerDataset(train_files, speaker_to_idx, train=True)
    val_ds = SpeakerDataset(val_files, speaker_to_idx, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0
    )

    print("Loading pretrained encoder...")
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa-voxceleb"
    ).to(DEVICE)
    encoder.eval()

    with torch.no_grad():
        dummy = torch.randn(2, SAMPLE_RATE, device=DEVICE)
        dummy_emb = encoder.encode_batch(dummy)
        if dummy_emb.dim() == 3:
            dummy_emb = dummy_emb.squeeze(1)
        emb_dim = dummy_emb.shape[-1]

    head = nn.Linear(emb_dim, num_speakers).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    best_val_loss = float("inf")
    bad_epochs = 0
    best_path = OUT_DIR / "best_checkpoint.pt"
    history = []

    for epoch in range(1, EPOCHS + 1):
        encoder.eval()
        head.train()

        tr_loss = 0.0
        tr_acc = 0.0
        n = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{EPOCHS}")
        for wavs, wav_lens, labels, _ in pbar:
            wavs = wavs.to(DEVICE, non_blocking=True)
            wav_lens = wav_lens.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            with torch.no_grad():
                emb = encoder.encode_batch(wavs, wav_lens=wav_lens)
                if emb.dim() == 3:
                    emb = emb.squeeze(1)

            logits = head(emb)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()

            bs = labels.size(0)
            tr_loss += loss.item() * bs
            tr_acc += batch_accuracy(logits, labels) * bs
            n += bs
            pbar.set_postfix(loss=loss.item())

        tr_loss /= max(1, n)
        tr_acc /= max(1, n)

        val_loss, val_acc = evaluate(encoder, head, val_loader, criterion)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"]
        })

        print(f"epoch={epoch} train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save({
                "head_state_dict": head.state_dict(),
                "speaker_to_idx": speaker_to_idx,
                "idx_to_speaker": {v: k for k, v in speaker_to_idx.items()},
                "emb_dim": emb_dim,
                "sample_rate": SAMPLE_RATE,
                "top_k": TOP_K,
                "config": {
                    "batch_size": BATCH_SIZE,
                    "epochs": EPOCHS,
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "val_ratio": VAL_RATIO,
                    "seed": SEED
                }
            }, best_path)
            print(f"saved: {best_path}")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("early stopping")
                break

    with open(OUT_DIR / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()