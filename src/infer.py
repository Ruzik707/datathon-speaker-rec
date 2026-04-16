import pandas as pd
import numpy as np
from pathlib import Path
from speechbrain.inference import EncoderClassifier
from speechbrain.dataio.dataio import read_audio
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# ==================== НАСТРОЙКИ ====================
TEST_PATH = Path(__file__).parent.parent / 'data' / 'test' / 'test_public'
K = 5

# ==================== 1. ЗАГРУЗКА МОДЕЛИ ====================
print("Загрузка модели для инференса...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec-ecapa-voxceleb"
)
# Переводим в режим оценки (выключаем dropout и т.д.)
classifier.eval()
print("Модель готова!")

# ==================== 2. ЗАГРУЗКА ТЕСТОВЫХ ФАЙЛОВ ====================
print("Поиск тестовых файлов...")
test_files = list(TEST_PATH.glob('**/*.flac'))

print(f"Найдено {len(test_files)} файлов для проверки.")

# ==================== 3. ПОЛУЧЕНИЕ ЭМБЕДДИНГОВ ====================
print("Извлечение эмбеддингов...")
embeddings = []

for file_path in tqdm(test_files):
    try:
        # 1. Читаем аудио
        signal = read_audio(str(file_path))

        # 2. Получаем вектор (эмбеддинг)
        # .detach().cpu().numpy() переводит тензор в обычный массив numpy
        embedding = classifier.encode_batch(signal)
        embedding = embedding.squeeze().detach().cpu().numpy()

        embeddings.append(embedding)
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {e}")

# Превращаем список векторов в матрицу [N_files, 192]
embeddings = np.array(embeddings)
print(f"Успешно обработано {len(embeddings)} файлов.")

# ==================== 4. ПОИСК СОСЕДЕЙ (KNN) ====================
print("Поиск ближайших соседей...")

# Считаем матрицу косинусного сходства (каждый с каждым)
# Результат: матрица размером [N, N], где sims[i][j] - похожеть файла i и j
sims = cosine_similarity(embeddings)

results = []

# Для каждого файла находим топ-K соседей
for i in tqdm(range(len(test_files))):
    # Получаем строку сходств для текущего файла i
    current_file_path = test_files[i]
    relative_path = "test_public/" + str(current_file_path.relative_to(TEST_PATH))

    # Сортируем индексы по убыванию сходства (от самого похожего)
    # argsort возвращает индексы от меньшего к большему, [::-1] разворачивает
    sorted_indices = np.argsort(sims[i])[::-1]

    # Берем первые K+1 индексов (так как 0-й индекс - это сам файл)
    # Нам нужны соседи, исключаем самого себя (индекс i)
    # Но sorted_indices[0] может быть не i, если есть файлы идентичные.
    # Безопасный способ: отфильтровать свой индекс

    neighbors = []
    count = 0
    for idx in sorted_indices:
        if idx == i:
            continue  # Пропускаем сам файл
        neighbors.append(idx)
        count += 1
        if count == K:
            break

    # Формируем строку для CSV: "5, 12, 88, 2, 10"
    neighbors_str = ",".join(map(str, neighbors))

    results.append([relative_path, neighbors_str])

# ==================== 5. СОХРАНЕНИЕ SUBMISSION ====================
print("Сохранение submission.csv...")

# Создаем DataFrame
df = pd.DataFrame(results, columns=['filepath', 'neighbours'])

# Сохраняем
df.to_csv('submission.csv', index=False)

print("Готово! Файл submission.csv создан.")
print("Пример первых строк:")
print(df.head())