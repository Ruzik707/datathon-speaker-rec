from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import faiss
from speechbrain.inference.classifiers import EncoderClassifier
from speechbrain.dataio.dataio import read_audio

warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "test"
TEST_CSV = ROOT / "data" / "test_public.csv"
CHECKPOINT_PATH = ROOT / "artifacts" / "best_checkpoint.pt"
OUTPUT_PATH = ROOT / "submission.csv"

BATCH_SIZE = 16
NUM_WORKERS = 0  # for debugging set 0, for faster inference try 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 10

# ===================== PATHS =====================
def resolve_audio_path(filepath: str) -> Path:
    p = Path(str(filepath))

    candidates = []

    # 1) as-is relative to DATA_ROOT
    candidates.append(DATA_ROOT / p)

    # 2) if path starts with test_public / test_private, strip first part
    if len(p.parts) > 1 and p.parts[0] in {"test_public", "test_private"}:
        candidates.append(DATA_ROOT / Path(*p.parts[1:]))

    # 3) fallback by filename only
    candidates.append(DATA_ROOT / p.name)
    candidates.append(ROOT / p)
    candidates.append(ROOT / p.name)

    seen = set()
    unique_candidates = []
    for c in candidates:
        c = c.resolve() if c.exists() else c
        if str(c) not in seen:
            seen.add(str(c))
            unique_candidates.append(c)

    for c in unique_candidates:
        if c.exists():
            return c

    raise FileNotFoundError(
        f"Audio file not found for filepath='{filepath}'. Tried: "
        + " | ".join(str(c) for c in unique_candidates)
    )

# ===================== DATASET =====================
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
        return wav, str(rel_path), str(full_path)

def collate_fn(batch):
    wavs, rel_paths, full_paths = zip(*batch)
    lengths = torch.tensor([w.shape[-1] for w in wavs], dtype=torch.float32)
    max_len = int(lengths.max().item())

    padded = []
    for w in wavs:
        if w.shape[-1] < max_len:
            w = torch.nn.functional.pad(w, (0, max_len - w.shape[-1]))
        padded.append(w)

    wavs = torch.stack(padded, dim=0)
    wav_lens = lengths / max_len
    return wavs, wav_lens, list(rel_paths), list(full_paths)

# ===================== EMBEDDINGS =====================
@torch.no_grad()
def embed_files(encoder, loader):
    encoder.eval()
    all_embs = []
    all_paths = []

    for wavs, wav_lens, rel_paths, _ in tqdm(loader, desc="Embedding"):
        wavs = wavs.to(DEVICE, non_blocking=True)
        wav_lens = wav_lens.to(DEVICE, non_blocking=True)

        emb = encoder.encode_batch(wavs, wav_lens=wav_lens)
        if emb.dim() == 3:
            emb = emb.squeeze(1)

        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        all_embs.append(emb.cpu())
        all_paths.extend(rel_paths)

    all_embs = torch.cat(all_embs, dim=0)
    return all_embs, all_paths

# ===================== SUBMISSION =====================
def load_test_list(test_csv_path: Path):
    df = pd.read_csv(test_csv_path)

    if "filepath" in df.columns:
        paths = df["filepath"].astype(str).tolist()
        col_name = "filepath"
    elif "Filepath" in df.columns:
        paths = df["Filepath"].astype(str).tolist()
        col_name = "Filepath"
    else:
        raise ValueError("test.csv must contain 'filepath' or 'Filepath' column")

    return df, paths, col_name

def build_submission(test_df, filepaths, neighbor_indices, out_path: Path):
    rows = []
    for path, neigh in zip(filepaths, neighbor_indices):
        neigh = [int(x) for x in neigh]
        neigh = list(dict.fromkeys(neigh))
        neigh_str = ",".join(map(str, neigh))
        rows.append({"Filepath": path, "Neighbours": neigh_str})

    sub = pd.DataFrame(rows)

    # preserve original order if test_df had filepath column
    if "filepath" in test_df.columns:
        sub["Filepath"] = test_df["filepath"].astype(str).tolist()
    elif "Filepath" in test_df.columns:
        sub["Filepath"] = test_df["Filepath"].astype(str).tolist()

    sub.to_csv(out_path, index=False, encoding="utf-8")

# ===================== MAIN =====================
def main():
    print("Loading test list...")
    test_df, test_paths, _ = load_test_list(TEST_CSV)

    print(f"Test items: {len(test_paths)}")
    for i, p in enumerate(test_paths[:3]):
        print(f"Sample path[{i}]: {p}")

    dataset = TestAudioDataset(test_paths)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(NUM_WORKERS > 0)
    )

    print("Loading checkpoint...")
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    emb_dim = ckpt.get("emb_dim", None)
    top_k_train = ckpt.get("top_k", TOP_K)
    if top_k_train != TOP_K:
        print(f"Warning: checkpoint top_k={top_k_train}, inference TOP_K={TOP_K}")

    print("Loading pretrained encoder...")
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(ROOT / "pretrained_models" / "spkrec-ecapa-voxceleb")
    ).to(DEVICE)
    encoder.eval()

    print("Building embeddings...")
    embeddings, rel_paths = embed_files(encoder, loader)
    embeddings_np = embeddings.numpy().astype(np.float32)

    if embeddings_np.shape[0] != len(rel_paths):
        raise RuntimeError("Embeddings count does not match paths count")

    print("Building FAISS index...")
    faiss.normalize_L2(embeddings_np)
    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)

    k_search = TOP_K + 1
    scores, neighbors = index.search(embeddings_np, k_search)

    print("Filtering self-match and duplicates...")
    final_neighbors = []
    for i, row in enumerate(neighbors):
        filtered = []
        seen = {i}
        for j in row:
            j = int(j)
            if j in seen:
                continue
            seen.add(j)
            filtered.append(j)
            if len(filtered) == TOP_K:
                break

        if len(filtered) < TOP_K:
            all_ids = list(range(len(rel_paths)))
            for j in all_ids:
                if j not in seen:
                    filtered.append(j)
                    seen.add(j)
                if len(filtered) == TOP_K:
                    break

        final_neighbors.append(filtered[:TOP_K])

    print("Saving submission...")
    build_submission(test_df, rel_paths, final_neighbors, OUTPUT_PATH)
    print(f"Done. Saved submission to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()