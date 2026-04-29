from pathlib import Path
import warnings
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import faiss
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.dataio.dataio import read_audio

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "test_public"
TEST_CSV = ROOT / "test_public.csv"
CHECKPOINT_PATH = ROOT / "artifacts" / "best_head.pt"
OUTPUT_PATH = ROOT / "submission.csv"

SAMPLE_RATE = 16000
VIEW_SECONDS = [3.0, 5.0, 8.0]
VIEW_SAMPLES = [int(SAMPLE_RATE * s) for s in VIEW_SECONDS]
VIEWS_PER_FILE = len(VIEW_SECONDS)

NOISE_P = 0.08
GAIN_P = 0.06
NOISE_LEVEL = 0.004
GAIN_MIN = 0.97
GAIN_MAX = 1.03

TOP_K = 10
BATCH_SIZE = 8
NUM_WORKERS = 2

QE_TOPK = 5
QE_ALPHA = 0.7

DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = torch.device(DEVICE_STR)
USE_AMP = torch.cuda.is_available()

def resolve_audio_path(filepath: str) -> Path:
    p = Path(str(filepath))
    candidates = [
        DATA_ROOT / p,
        DATA_ROOT / Path(*p.parts[1:]) if len(p.parts) > 1 and p.parts[0] in {"test_public", "test_private"} else None,
        ROOT / p,
        ROOT / p.name,
    ]
    for c in candidates:
        if c is None:
            continue
        if c.exists():
            return c
    raise FileNotFoundError(f"Audio file not found: {filepath}")

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

def build_views(wav):
    views = []
    for sec in VIEW_SECONDS:
        seg = crop_random(wav, int(SAMPLE_RATE * sec))
        seg = maybe_noise(seg)
        seg = maybe_gain(seg)
        views.append(seg)
    return views

class TestAudioDataset(Dataset):
    def __init__(self, filepaths):
        self.filepaths = list(filepaths)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        rel_path = self.filepaths[idx]
        full_path = resolve_audio_path(rel_path)
        wav = read_audio(str(full_path))
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        elif wav.dim() > 2:
            wav = wav.squeeze()
        wav = wav.float()
        return build_views(wav), str(rel_path)

def collate_fn(batch):
    all_segs = []
    all_paths = []
    counts = []

    for views, path in batch:
        counts.append(len(views))
        for seg in views:
            all_segs.append(seg)
            all_paths.append(path)

    lengths = torch.tensor([s.shape[-1] for s in all_segs], dtype=torch.float32)
    max_len = int(lengths.max().item())
    padded = []
    for s in all_segs:
        if s.shape[-1] < max_len:
            s = torch.nn.functional.pad(s, (0, max_len - s.shape[-1]))
        padded.append(s)

    wavs = torch.stack(padded, dim=0)
    wav_lens = lengths / max_len
    return wavs, wav_lens, all_paths, counts

def load_test_list(test_csv_path: Path):
    df = pd.read_csv(test_csv_path)
    if "filepath" in df.columns:
        return df, df["filepath"].astype(str).tolist()
    if "Filepath" in df.columns:
        return df, df["Filepath"].astype(str).tolist()
    raise ValueError("test.csv must contain filepath or Filepath column")

@torch.no_grad()
def embed_test(encoder, loader):
    encoder.eval()
    embs = []
    paths = []

    for wavs, wav_lens, batch_paths, counts in tqdm(loader, desc="Embedding"):
        wavs = wavs.to(DEVICE, non_blocking=True)
        wav_lens = wav_lens.to(DEVICE, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=USE_AMP):
            out = encoder.encode_batch(wavs, wav_lens=wav_lens)

        if out.dim() == 3:
            out = out.squeeze(1)

        out = torch.nn.functional.normalize(out, p=2, dim=-1).cpu()

        idx = 0
        for p in batch_paths[::VIEWS_PER_FILE]:
            vecs = out[idx:idx + VIEWS_PER_FILE]
            v = vecs.mean(dim=0)
            v = torch.nn.functional.normalize(v, p=2, dim=0)
            embs.append(v.unsqueeze(0))
            paths.append(p)
            idx += VIEWS_PER_FILE

    return torch.cat(embs, dim=0), paths

def average_query_expansion(query_vecs, ref_vecs, top_k=5, alpha=0.7):
    q = query_vecs.astype(np.float32).copy()
    r = ref_vecs.astype(np.float32).copy()

    faiss.normalize_L2(q)
    faiss.normalize_L2(r)

    index = faiss.IndexFlatIP(r.shape[1])
    index.add(r)
    sims, idx = index.search(q, top_k)

    weights = np.power(np.maximum(sims, 0.0), alpha).astype(np.float32)
    weights_sum = np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
    weights = weights / weights_sum

    expanded = []
    for i in range(q.shape[0]):
        neigh = r[idx[i]]
        mean_vec = (neigh * weights[i][:, None]).sum(axis=0)
        new_q = q[i] + 0.5 * mean_vec
        expanded.append(new_q)

    expanded = np.stack(expanded, axis=0).astype(np.float32)
    faiss.normalize_L2(expanded)
    return expanded

def save_submission(order_paths, neighbors, out_path: Path):
    rows = []
    for path, neigh in zip(order_paths, neighbors):
        neigh = [int(x) for x in neigh]
        neigh = list(dict.fromkeys(neigh))
        rows.append({"Filepath": path, "Neighbours": ",".join(map(str, neigh[:TOP_K]))})
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")

def main():
    random.seed(42)

    _, test_paths = load_test_list(TEST_CSV)
    dataset = TestAudioDataset(test_paths)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0
    )

    _ = torch.load(CHECKPOINT_PATH, map_location="cpu")

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb"),
        run_opts={"device": DEVICE_STR},
    )

    embs, order_paths = embed_test(encoder, loader)
    x = embs.numpy().astype(np.float32)
    faiss.normalize_L2(x)

    index = faiss.IndexFlatIP(x.shape[1])
    index.add(x)

    qe_x = average_query_expansion(x, x, top_k=QE_TOPK, alpha=QE_ALPHA)

    _, neigh1 = index.search(x, TOP_K + 1)
    _, neigh2 = index.search(qe_x, TOP_K + 1)

    final_neighbors = []
    for i in range(len(x)):
        merged = []
        seen = {i}
        for row in (neigh1[i], neigh2[i]):
            for j in row:
                j = int(j)
                if j in seen:
                    continue
                seen.add(j)
                merged.append(j)
                if len(merged) == TOP_K:
                    break
            if len(merged) == TOP_K:
                break
        final_neighbors.append(merged)

    save_submission(order_paths, final_neighbors, OUTPUT_PATH)
    print(f"Saved submission to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()