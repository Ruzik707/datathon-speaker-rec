# Криптонит.Тембр — Speaker Recognition Datathon

Хакатонный прототип для аудио-retrieval: для каждой записи найти 10 наиболее
похожих записей того же говорящего и сформировать `submission.csv`.

**Мой вклад в команде:** разбор задачи, собственный baseline на SpeechBrain
ECAPA-TDNN, извлечение эмбеддингов, cosine similarity / nearest-neighbour search,
формирование submission и эксперименты с линейной головой и шумовыми аугментациями.

**Стек:** Python 3.12+, PyTorch, SpeechBrain, NumPy, Pandas, FAISS.
Итоговая конкурсная метрика в репозитории не зафиксирована; улучшение качества
от экспериментальных вариантов не заявляется.

## Пайплайн и варианты

```text
Аудио → ECAPA-TDNN → L2-нормализация → FAISS IndexFlatIP
      → исключение самой записи и дублей → top-10 → CSV
```

| Вариант | Где смотреть | Данные относительно корня репозитория |
| --- | --- | --- |
| Базовый retrieval | [src/infer.py](src/infer.py) | `data/test_public.csv`, аудио в `data/test/` |
| Эксперимент с окнами 3/5/8 секунд и α-QE | [src_gpu/infer.py](src_gpu/infer.py) | `src_gpu/test_public.csv`, аудио в `src_gpu/test_public/` |
| Обучение головы | [src/train.py](src/train.py), [src_gpu/train_head.py](src_gpu/train_head.py) | отдельный обучающий набор |

Голова — отдельный классификационный эксперимент. Retrieval использует
предобученный энкодер, поэтому checkpoint головы для инференса не требуется.

## Запуск базового варианта

```bash
git clone https://github.com/Ruzik707/datathon-speaker-rec.git
cd datathon-speaker-rec
uv sync
```

Нужны аудиоданные соревнования — в Git они не включены:

```text
data/
├── test_public.csv         # колонка filepath или Filepath
└── test/                   # FLAC-файлы, соответствующие строкам CSV
```

```bash
uv run python src/infer.py
# Результат: submission.csv в корне проекта
```

Энкодер `speechbrain/spkrec-ecapa-voxceleb` загружается при первом запуске.
CUDA используется при наличии; для CPU обработка большого набора будет медленной.
Используйте совместимые сборки PyTorch и torchaudio для своей платформы.

## Дополнительный эксперимент

Для multi-view варианта положите CSV и аудио в пути из таблицы, затем:

```bash
uv run python src_gpu/infer.py
# Результат: src_gpu/submission.csv
```

[Предвычисление train-эмбеддингов](src_gpu/precompute_embeddings.py) и
[обучение головы](src_gpu/train_head.py) запускаются отдельно и требуют
`src_gpu/train_part_1/train/<speaker_id>/...` — `src_gpu/train_part_10/train/...`.
Материалы хакатона: [отчёт](src_gpu/report.docx), [презентация](src_gpu/presentation.pptx).

## Формат результата и ограничения

`Filepath,Neighbours`: исходный путь и список 10 индексов через запятую.
Индекс самой записи и повторения исключаются; для top-10 нужны минимум 11 записей.

- Аудио ожидается в частоте модели, 16 кГц; явного ресемплинга в скриптах нет.
- Multi-view использует случайные crops и аугментации. Полная детерминированность
  при смене числа workers и устройства не гарантируется.
- В ветке α-QE списки сливаются последовательно: если исходный top-10 уже заполнен,
  соседи расширенного запроса не меняют результат. Польза QE не подтверждена.
- Набор аудио и оценка Precision@K не воспроизводятся одним клонированием репозитория.
