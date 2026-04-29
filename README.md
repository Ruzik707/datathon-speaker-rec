# Криптонит.Тембр — Speaker Recognition

Решение задачи распознавания диктора (nearest-neighbour retrieval) на базе
предобученного энкодера **SpeechBrain ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`)**
с пост-обработкой через **Average Query Expansion (α-QE)** и многокадровую
агрегацию (multi-view TTA) поверх **FAISS IndexFlatIP**.

Для каждого тестового файла формируется список **10 ближайших соседей** в
формате `submission.csv` (метрика соревнования — **Precision@K**).

---

## 1. Окружение

- **OS:** Ubuntu 20.04+ / Windows 10+ (решение протестировано под обе платформы)
- **Python:** 3.10–3.13
- **GPU:** рекомендуется Nvidia с CUDA ≥ 12.x (решение совместимо с A40 / 45 GB)
- **CPU-режим** поддерживается автоматически (fallback без AMP).

Установка зависимостей:

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install --upgrade pip
pip install -r requirements.txt
```

> Перед первым запуском SpeechBrain автоматически скачает веса
> `speechbrain/spkrec-ecapa-voxceleb` в папку
> `pretrained_models/spkrec-ecapa-voxceleb/` (в корне репозитория).

---

## 2. Ожидаемая структура каталогов

```
submission_repo/
├── precompute_embeddings.py     # 1) извлечение train-эмбеддингов
├── train_head.py                # 2) обучение линейной головы
├── infer.py                     # 3) инференс -> submission.csv
├── requirements.txt
├── README.md
├── report.docx                  # отчёт по работе
├── pretrained_models/
│   └── spkrec-ecapa-voxceleb/   # подгружается автоматически
├── artifacts/                   # результаты обучения
│   ├── train_embeddings.pt
│   ├── train_labels.pt
│   ├── speaker_to_idx.json
│   ├── meta.json
│   ├── best_head.pt
│   └── head_history.json
├── train_part_1/train/<speaker_id>/*.flac
├── ...
├── train_part_10/train/<speaker_id>/*.flac
├── test_public/                 # тестовые FLAC-записи
├── test_public.csv              # колонка `filepath` (как в шаблоне)
└── submission.csv               # формируется инференсом
```

Требования к `test_public.csv`:
- обязательная колонка `filepath` (или `Filepath`);
- пути к файлам — ровно те, что указаны в шаблоне (например,
  `test_public/000000.flac`) — они попадают в итоговый `submission.csv` без изменений.

---

## 3. Запуск «под ключ»

### 3.1. Инференс (если веса уже лежат в `artifacts/`)

```bash
python infer.py
```

Скрипт:
1. читает `test_public.csv`,
2. строит для каждого FLAC-файла multi-view эмбеддинг (3 / 5 / 8 сек окна),
3. индексирует эмбеддинги в FAISS (`IndexFlatIP` — эквивалент cosine
   similarity благодаря L2-нормировке),
4. применяет Average Query Expansion (top-5, α = 0.7),
5. объединяет два списка соседей (original + QE) в один топ-10,
6. сохраняет `submission.csv` в формате
   `Filepath, Neighbours` (10 индексов через запятую).

### 3.2. Полный пайплайн с нуля

```bash
# (1) извлечь эмбеддинги обучающей выборки и мета-данные
python precompute_embeddings.py

# (2) обучить линейную голову (используется как sanity-check
#     качества эмбеддингов на speaker classification)
python train_head.py

# (3) получить финальный submission.csv
python infer.py
```

---

## 4. Формат выхода

`submission.csv` — UTF-8, разделитель `,`:

```
Filepath,Neighbours
test_public/000000.flac,"1437,1456,120550,109576,114835,103742,79819,20026,22920,102307"
...
```

- ровно одна строка на тестовую запись;
- `Filepath` — в точности из шаблона `test_public.csv`;
- `Neighbours` — 10 целых индексов через запятую, **без дублей**, **без
  индекса самой записи**, **без NaN / пропусков**.

---

## 5. Веса модели

- Энкодер: `speechbrain/spkrec-ecapa-voxceleb` (Apache 2.0, публичный HF
  репозиторий). Скачивается автоматически в `pretrained_models/`.
- Голова: `artifacts/best_head.pt` — формируется `train_head.py`.

VoxBlink2 и его производные **не используются** (см. правила соревнования).

---

## 6. Воспроизводимость

- Глобальные seed = 42 (`random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`).
- `torch.backends.cudnn.benchmark = True` (для скорости; при строгой
  детерминированности рекомендуется отключить).
- AMP включается автоматически при наличии CUDA.
