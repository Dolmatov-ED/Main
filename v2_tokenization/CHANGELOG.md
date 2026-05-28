# v2 — Hybrid Tokenization & Spatial Feature Engineering

## Overview
Добавлен модуль гибридной токенизации поверх ETL-пайплайна v1.
Непрерывные признаки → MLP-проекторы → d_model эмбеддинги.
Дискретные события → vocabulary lookup.
Слои карты → MapTensor для будущего Map-AE.

## Added
- `tokenizer/projectors.py` — ContinuousProjector, PositionProjector, OrientProjector, StateProjector, CoverProjector
- `tokenizer/events.py` — EventEmbedder, EVENT_VOCAB (20+ токенов)
- `tokenizer/hybrid.py` — HybridTokenizer: сумма проекций + LayerNorm → [B, S, d_model]
- `tokenizer/map_layers.py` — MapLayerGenerator (Height, Walkability, Cover_Score)
- `mocks/mock_tokenizer.py` — генераторы синтетических тензоров (BatchGenerator)
- `tests/test_tokenizer.py` — изолированные тесты для токенизатора

## Architecture
```
[Raw State Vector] → (Continuous MLPs) → [Emb_Pos + Emb_Orient + Emb_State + Emb_Cover]
[Event IDs]        → (Lookup Table)     → [Emb_Event]
                                      ↓
                        [Sum + LayerNorm] → [Token ∈ ℝ^{d_model}]
```

## Key design decisions
- MLP с GELU вместо ReLU (гладкие градиенты)
- Dimension: d_model = 512 (конфигурируемо)
- Проекторы: Position(5→64→512), Orient(3→32→512), State(3→32→512), Cover(1→16→512)
- Словарь событий: 256 токенов с padding_idx=0
- MapTensor: [C=3, H=256, W=256] (Height, Walkability, Cover)
- Все моки генерируют float32 тензоры, совместимые с PyTorch

## Dependencies added
- torch >= 2.0
- numpy, pandas (from v1)

## Inherits from v1
- `etl/` модули (parser, aligner, segmenter, exporter, validators)
- `mocks/mock_demo.py` (MockDemo, MockDemoParser)
