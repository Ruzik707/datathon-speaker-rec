from pathlib import Path
from speechbrain.inference import EncoderClassifier
from speechbrain.dataio.dataio import read_audio
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from tqdm import tqdm

DATA_PATH = Path(__file__).parent.parent / 'data' / 'raw' / 'train_part_1' / 'train'
K = 5


# ==================== АУГМЕНТАЦИЯ ЧЕРЕЗ TORCHAUDIO ====================
class AudioAugmentor:
    """Аугментация с использованием torchaudio"""

    def __init__(self, noise_level=0.1, reverb=False):
        self.noise_level = noise_level
        self.reverb = reverb

    def add_noise(self, signal):
        """Добавляет сильный белый шум"""
        noise = torch.randn_like(signal) * self.noise_level
        return signal + noise

    def add_reverb(self, signal):
        """Имитация реверберации через свертку"""
        # Простая имитация: несколько эхо с затуханием
        reverbed = signal.clone()
        for delay in [5000, 10000, 15000]:  # задержки в сэмплах
            if signal.shape[-1] > delay:
                echo = signal[..., :-delay] * 0.3
                reverbed[..., delay:delay + echo.shape[-1]] += echo
        return reverbed

    def perturb_speed(self, signal, sample_rate=16000):
        """Изменение скорости воспроизведения"""
        factor = np.random.choice([0.9, 1.0, 1.1])  # -10%, 0%, +10%
        if factor == 1.0:
            return signal
        # Простая реализация через ресемплинг
        length = int(signal.shape[-1] * factor)
        if length > signal.shape[-1]:
            length = signal.shape[-1]
        indices = torch.linspace(0, signal.shape[-1] - 1, length).long()
        return signal[..., :length] if length < signal.shape[-1] else signal

    def augment(self, signal):
        """Применяет все аугментации последовательно"""
        # Сильный шум
        signal = self.add_noise(signal)

        # Реверберация
        if self.reverb:
            signal = self.add_reverb(signal)

        # Изменение скорости
        signal = self.perturb_speed(signal)

        return signal


# ==================== 1. ЗАГРУЗКА ДАННЫХ ====================
print("Загрузка данных...")
audio_files = list(DATA_PATH.glob('**/*.flac'))
print(f"Найдено файлов: {len(audio_files)}")

file_to_speaker = {}
for file_path in audio_files:
    speaker_id = file_path.parent.name
    file_to_speaker[str(file_path)] = speaker_id

speakers = list(set(file_to_speaker.values()))
print(f"Количество дикторов: {len(speakers)}")

# ==================== 2. ЗАГРУЗКА МОДЕЛИ ====================
print("\nЗагрузка модели...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)

# ==================== 3. АУГМЕНТИРОВАННИЕ И ПОЛУЧЕНИЕ ЭМБЕДДИНГОВ ====================
# Создаем аугментор с СИЛЬНЫМИ искажениями
augmentor = AudioAugmentor(noise_level=0.2, reverb=True)


def get_embeddings(augmentor=None, max_files=500):
    """Получает эмбеддинги с опциональной аугментацией"""
    embeddings = []
    paths = []

    for file_path in tqdm(audio_files[:max_files], desc="Обработка"):
        try:
            signal = read_audio(str(file_path))

            if augmentor is not None:
                signal = augmentor.augment(signal)

            embedding = classifier.encode_batch(signal)
            embedding = embedding.squeeze().detach().cpu().numpy()
            embeddings.append(embedding)
            paths.append(str(file_path))
        except Exception as e:
            print(f"Ошибка {file_path}: {e}")

    return np.array(embeddings), paths

# Получаем эмбеддинги для аугментированных данных
print("\nПолучение эмбеддингов аугментированных данных (шум + ревербация)")
embeddings_aug, paths_aug = get_embeddings(augmentor=augmentor, max_files=5000)


def calculate_precision(embeddings, paths, k=5):
    """Считает Precision@K"""
    sims = cosine_similarity(embeddings)
    scores = []

    for i in range(len(paths)):
        current_speaker = file_to_speaker[paths[i]]
        sorted_indices = np.argsort(sims[i])[::-1]
        top_k_indices = sorted_indices[1:k + 1]

        relevant = sum(1 for idx in top_k_indices
                       if file_to_speaker[paths[idx]] == current_speaker)
        scores.append(relevant / k)

    return np.mean(scores)


# Считаем метрику
precision_aug = calculate_precision(embeddings_aug, paths_aug, K)

print("РЕЗУЛЬТАТЫ:")
print("=" * 60)
print(f"Precision@{K}: {precision_aug:.4f}")