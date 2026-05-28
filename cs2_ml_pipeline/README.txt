# CS2 ML Pipeline
## Индикатор оптимальности игровых действий в Counter-Strike 2 на основе методов машинного обучения

Полный пайплайн анализа демо-файлов CS2: парсинг → токенизация → модель → дашборд.

---

## Требования

- **Python 3.11–3.12** (рекомендуется 3.12)
- **pip** (менеджер пакетов Python)
- **ОС:** Windows 10/11, Linux (Ubuntu 20.04+), macOS (Apple Silicon)
- **GPU (опционально):** NVIDIA с CUDA 11.8+ для ускорения обучения
- **Диск:** ~500 МБ для модели и временных файлов

---

## Как запустить

### 1. Клонируйте проект или скопируйте папку
```bash
cd cs2_ml_pipeline
```

### 2. Установите зависимости
```bash
pip install -r requirements.txt
```

### 3. Запустите пайплайн

**Быстрый старт (синтетические данные, без реальной демки):**
```bash
python main.py --demo-mode --train --dashboard
```

**С реальной демкой:**
```bash
python main.py --demo path/to/match.dem --train --dashboard
```

**Только ETL (экспорт данных в Parquet):**
```bash
python main.py --demo path/to/match.dem
```

**Пакетное обучение на папке с демками:**
```bash
python auto_train.py --dir D:\demos --epochs 10 --keep
```

---

## Параметры командной строки

| Параметр | По умолчанию | Описание |
|----------|-------------|----------|
| `--demo` | — | Путь к .dem / .dem.zst / .dem.gz файлу |
| `--demo-mode` | False | Использовать синтетические данные |
| `--train` | False | Запустить обучение модели |
| `--dashboard` | False | Показать дашборд после обучения |
| `--epochs` | 10 | Количество эпох обучения |
| `--d-model` | 256 | Размерность модели (128–768) |
| `--n-layers` | 8 | Количество слоёв Transformer |
| `--n-heads` | 4 | Количество голов внимания |
| `--seq-len` | 128 | Длина последовательности |
| `--lr` | 3e-4 | Learning rate |
| `--target-hz` | 8 | Частота дискретизации (Гц) |
| `--cpu` | False | Использовать CPU вместо GPU |
| `--output` | output | Папка для результатов |
| `--player` | — | Фильтр по имени игрока |
| `--seed` | 42 | Random seed |

---

## Структура проекта

```
cs2_ml_pipeline/
├── cs2_ml_pipeline/        # Пакет с исходным кодом
│   ├── etl/                # Парсинг .dem, выравнивание, экспорт
│   ├── tokenizer/          # Гибридный токенизатор
│   ├── models/             # Transformer, Map-AE, heads
│   ├── training/           # Тренер, curriculum, contrastive
│   └── inference/          # Стриминг, дашборд, XAI, ONNX
├── main.py                 # Точка входа
├── auto_train.py           # Пакетное обучение
├── requirements.txt        # Зависимости
└── README.md               # Этот файл
```

---

## Примечания

- При первом запуске с реальной демкой может потребоваться `pip install zstandard` для распаковки `.zst` файлов.
- Для GPU-ускорения убедитесь, что установлен PyTorch с CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- Синтетический режим (`--demo-mode`) не требует файлов `.dem` и подходит для тестирования.
