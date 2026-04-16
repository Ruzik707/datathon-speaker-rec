from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from speechbrain.inference import EncoderClassifier
from speechbrain.dataio.dataio import read_audio
from torch.utils.data import DataLoader, Dataset
from speechbrain.augment.time_domain import SpeedPerturb
import torchaudio.functional as F
from tqdm import tqdm
import random


# ==================== НАСТРОЙКИ ====================
DATA_PATH = Path(__file__).parent.parent / 'data' / 'raw' / 'train_part_1' / 'train'
K = 5
BATCH_SIZE = 10
EPOCHS = 5
LEARNING_RATE = 1e-4
NUM_FILES_TO_TRAIN = None

# ==================== АУГМЕНТАЦИЯ ====================
class AudioAugmentor:
    def __init__(self, noise_level=0.15):
        self.noise_level = noise_level

    def add_noise(self, signal):
        noise = torch.randn_like(signal) * self.noise_level
        return signal + noise

    def augment(self, signal):
        return self.add_noise(signal)

# ==================== 1. ПОДГОТОВКА ДАННЫХ ====================
print("Загрузка списка файлов...")
audio_files = list(DATA_PATH.glob('**/*.flac'))
train_files = audio_files[:NUM_FILES_TO_TRAIN]

file_to_speaker = {}
speakers_list = []
for file_path in train_files:
    speaker_id = file_path.parent.name
    file_to_speaker[str(file_path)] = speaker_id
    if speaker_id not in speakers_list:
        speakers_list.append(speaker_id)

speaker_to_idx = {spk: i for i, spk in enumerate(speakers_list)}
num_speakers = len(speakers_list)
print(f"Дикторов: {num_speakers}, Файлов: {len(train_files)}")

# class AudioDataset(Dataset):
#     def __init__(self, file_paths, speaker_map, transform=None):
#         self.file_paths = file_paths
#         self.speaker_map = speaker_map
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.file_paths)
#
#     def __getitem__(self, idx):
#         file_path = self.file_paths[idx]
#         label = self.speaker_map[file_path]
#         signal = read_audio(str(file_path))
#         if self.transform:
#             signal = self.transform(signal)
#         return signal, label
#
# augmentor = AudioAugmentor(noise_level=0.15)
# dataset = AudioDataset(train_files, file_to_speaker, transform=augmentor.augment)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# ==================== 2. ЗАГРУЗКА МОДЕЛИ И НАСТРОЙКА ====================
print("Загрузка модели...")

base_classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# Создаем классификатор поверх эмбеддингов
# Размер эмбеддинга у ECAPA-TDNN обычно 192
embedding_dim = 192
output_layer = nn.Linear(embedding_dim, num_speakers)

# Оптимизатор: учим верхний слой
params_to_optimize = list(output_layer.parameters())
optimizer = optim.Adam(params_to_optimize, lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

augmentor = AudioAugmentor(noise_level=0.15)

# ==================== 3. ЦИКЛ ОБУЧЕНИЯ ====================
print(f"\nНачинаем дообучение (Epochs: {EPOCHS})...")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    random.shuffle(train_files)

    for i in tqdm(range(0, len(train_files), BATCH_SIZE), desc="Batch"):
        batch_files = train_files[i: i + BATCH_SIZE]
        if len(batch_files) < 2: break

        signals = []
        labels = []

        for f_path in batch_files:
            try:
                sig = read_audio(str(f_path))
                sig = augmentor.augment(sig)

                if sig.dim() == 1: sig = sig.unsqueeze(0)
                signals.append(sig)
                labels.append(speaker_to_idx[file_to_speaker[str(f_path)]])
            except:
                continue

        if not signals: continue

        # Простой паддинг (обрезаем до мин. длины)
        min_len = min(s.shape[-1] for s in signals)
        signals = [s[..., :min_len] for s in signals]

        x = torch.cat(signals, dim=0)
        y = torch.tensor(labels, dtype=torch.long)

        # Forward pass
        # Получаем эмбеддинги
        with torch.no_grad():  # Сначала без градиентов для энкодера (если он есть)
            embeddings = base_classifier.encode_batch(x)

        embeddings = embeddings.squeeze()

        logits = output_layer(embeddings)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Loss: {loss.item():.4f}")

# Сохраняем ТОЛЬКО верхний слой
torch.save(output_layer.state_dict(), "fine_tuned_head.pt")
print("\nГолова модели сохранена!")