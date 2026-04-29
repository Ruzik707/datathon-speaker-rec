from pathlib import Path
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.dataio.dataio import read_audio

ROOT = Path(__file__).resolve().parent
PARTS = [ROOT / f"train_part_{i}" / "train" for i in range(1, 11)]
OUT_DIR = ROOT / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
SHORT_SEC = 5.0
LONG_SEC = 8.0
LONG_THRESHOLD_SEC = 12.0

NOISE_P = 0.08
GAIN_P = 0.06
NOISE_LEVEL = 0.005
GAIN_MIN = 0.97
GAIN_MAX = 1.03

BATCH_SIZE = 12
NUM_WORKERS = 4
SEED = 42

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)
USE_AMP = torch.cuda.is_available()

def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

def crop_random(wav, crop_samples):
    n = wav.numel()
    if n <= crop_samples:
        return wav
    start = random.randint(0, n - crop_samples)
    return wav[start:start + crop_samples]

def maybe_noise(wav):
    if random.random() < NOISE_P:
        std = wav.std().clamp_min(1e-6)
        wav = wav + torch.randn_like(wav) * std * NOISE_LEVEL
    return wav

def maybe_gain(wav):
    if random.random() < GAIN_P:
        wav = wav * random.uniform(GAIN_MIN, GAIN_MAX)
    return wav

def make_views(wav):
    dur = wav.numel() / SAMPLE_RATE
    if dur < SHORT_SEC:
        return [wav]
    if dur < LONG_THRESHOLD_SEC:
        seg = crop_random(wav, int(SAMPLE_RATE * SHORT_SEC))
        return [maybe_gain(maybe_noise(seg))]
    seg1 = crop_random(wav, int(SAMPLE_RATE * SHORT_SEC))
    seg2 = crop_random(wav, int(SAMPLE_RATE * LONG_SEC))
    seg1 = maybe_gain(maybe_noise(seg1))
    seg2 = maybe_gain(maybe_noise(seg2))
    return [seg1, seg2]

class AudioDataset(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        wav = read_audio(str(path))
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        elif wav.dim() > 2:
            wav = wav.squeeze()
        wav = wav.float()
        views = make_views(wav)
        return views, label, str(path)

def collate_fn(batch):
    segs_all = []
    labels_all = []
    paths_all = []
    counts = []

    for views, label, path in batch:
        counts.append(len(views))
        for seg in views:
            segs_all.append(seg)
            labels_all.append(label)
            paths_all.append(path)

    lengths = torch.tensor([s.shape[-1] for s in segs_all], dtype=torch.float32)
    max_len = int(lengths.max().item())
    padded = []
    for s in segs_all:
        if s.shape[-1] < max_len:
            s = torch.nn.functional.pad(s, (0, max_len - s.shape[-1]))
        padded.append(s)

    wavs = torch.stack(padded, dim=0)
    wav_lens = lengths / max_len
    labels = torch.tensor(labels_all, dtype=torch.long)
    return wavs, wav_lens, labels, paths_all, counts

@torch.no_grad()
def extract_embeddings(encoder, loader):
    encoder.eval()
    file_embs = []
    file_labels = []
    file_paths = []

    for wavs, wav_lens, labels, paths, counts in tqdm(loader, desc="Embedding"):
        wavs = wavs.to(DEVICE, non_blocking=True)
        wav_lens = wav_lens.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            out = encoder.encode_batch(wavs, wav_lens=wav_lens)

        if out.dim() == 3:
            out = out.squeeze(1)

        out = torch.nn.functional.normalize(out, p=2, dim=-1).cpu()

        idx = 0
        for c in counts:
            vecs = out[idx:idx + c]
            v = vecs.mean(dim=0)
            v = torch.nn.functional.normalize(v, p=2, dim=0)
            file_embs.append(v.unsqueeze(0))
            file_labels.append(labels[idx].item())
            file_paths.append(paths[idx])
            idx += c

    return torch.cat(file_embs, dim=0), torch.tensor(file_labels, dtype=torch.long), file_paths

def main():
    items = []
    for part in PARTS:
        if not part.exists():
            continue
        for fp in sorted(part.glob("**/*.flac")):
            items.append((fp, fp.parent.name))

    speakers = sorted({spk for _, spk in items})
    speaker_to_idx = {s: i for i, s in enumerate(speakers)}
    indexed_items = [(fp, speaker_to_idx[spk]) for fp, spk in items]

    ds = AudioDataset(indexed_items)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0
    )

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"),
        run_opts={"device": DEVICE_STR},
    )

    with torch.no_grad():
        dummy = torch.randn(2, SAMPLE_RATE, device=DEVICE)
        dummy_emb = encoder.encode_batch(dummy)
        if dummy_emb.dim() == 3:
            dummy_emb = dummy_emb.squeeze(1)
        emb_dim = int(dummy_emb.shape[-1])

    embs, labels, paths = extract_embeddings(encoder, loader)

    torch.save(embs, OUT_DIR / "train_embeddings.pt")
    torch.save(labels, OUT_DIR / "train_labels.pt")

    with open(OUT_DIR / "speaker_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(speaker_to_idx, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "emb_dim": emb_dim,
            "sample_rate": SAMPLE_RATE,
            "num_items": len(paths),
            "num_speakers": len(speakers),
            "mode": "adaptive_1or2_views",
            "short_sec": SHORT_SEC,
            "long_sec": LONG_SEC
        }, f, ensure_ascii=False, indent=2)

    print(f"Saved embeddings to {OUT_DIR}")

if __name__ == "__main__":
    main()