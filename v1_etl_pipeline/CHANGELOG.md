# v1 — ETL & Data Pipeline Refactoring

## Overview
Модульный ETL-пайплайн, выделенный из cs2.py. Убирает matplotlib/tkinter зависимости,
добавляет downsampling 4-8 Гц, нормализацию координат/углов, экспорт в Parquet.

## Added
- `etl/parser.py` — низкоуровневый парсинг `.dem` (awpy + demoparser2)
- `etl/aligner.py` — слияние тиков, C4, гранат, событий
- `etl/segmenter.py` — сегментация на раунды, t_round, привязка событий
- `etl/exporter.py` — downsampling до target_hz, нормализация (cos/sin для yaw), экспорт в Parquet
- `etl/validators.py` — автоматические sanity checks
- `mocks/mock_demo.py` — MockDemo, MockDemoParser, генераторы синтетических данных
- `tests/` — изолированные тесты для каждого модуля

## Changed
- Полностью убран matplotlib/tkinter UI
- Координаты: абсолютные + относительные (dx_to_c4, dy_to_c4)
- Углы: yaw → cos(yaw)/sin(yaw)
- Частота: фиксированный ::2 → target_hz 4-8 Гц

## Removed
- InteractiveViewer, Slider, Button, plt.rcParams, tkinter
- Фиксированные MAP_CONFIGS для пиксельной конвертации
- _world_to_pixel, create_hex_layer, toggle_layer_btn

## Dependencies
- pandas, numpy, pyarrow (Parquet)
- Моки: awpy, demoparser2 замоканы для тестов
