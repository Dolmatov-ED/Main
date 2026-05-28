"""
auto_train.py — Обучает модель на всех .dem файлах из папки.
Инкрементально: парсинг → обучение → чекпоинт → следующий.

Usage:
    python auto_train.py                        # ищет .dem в текущей папке
    python auto_train.py --dir d:\\demos        # или в указанной папке
    python auto_train.py --d-model 512          # с изменёнными параметрами
    python auto_train.py --keep                 # не удалять демки после обработки
"""

import os, sys, glob, subprocess, argparse
from pathlib import Path

# ── Параметры по умолчанию (согласованы с main.py) ──
DEFAULT_D_MODEL = 256
DEFAULT_N_LAYERS = 8
DEFAULT_N_HEADS = 4
DEFAULT_EPOCHS = 5
DEFAULT_SEQ_LEN = 128
DEFAULT_LR = 3e-4
DEFAULT_SEED = 42
DEFAULT_TARGET_HZ = 8
OUTPUT_DIR = "output"


def main():
    parser = argparse.ArgumentParser(description="Обучить модель на папке с демками (инкрементально)")
    parser.add_argument("--dir", type=str, default=".",
                        help="папка с .dem файлами (по умолчанию текущая)")
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL,
                        help="размерность модели (128–768)")
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS,
                        help="количество слоёв Transformer (4–24)")
    parser.add_argument("--n-heads", type=int, default=DEFAULT_N_HEADS,
                        help="количество голов внимания (4–16)")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="эпох на каждую демку")
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN,
                        help="длина последовательности (64–1024)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR,
                        help="learning rate (1e-4 – 1e-2)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="random seed для воспроизводимости")
    parser.add_argument("--target-hz", type=int, default=DEFAULT_TARGET_HZ,
                        help="целевая частота дискретизации (4–16 Гц)")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR,
                        help="папка для чекпоинтов")
    parser.add_argument("--keep", action="store_true",
                        help="НЕ удалять демки после обучения (рекомендуется)")
    parser.add_argument("--cpu", action="store_true",
                        help="использовать CPU вместо GPU")
    parser.add_argument("--dashboard", action="store_true",
                        help="показать дашборд после обработки каждой демки")
    args = parser.parse_args()

    demos_dir = Path(args.dir)
    if not demos_dir.exists():
        print(f"[!] Папка не найдена: {demos_dir}")
        sys.exit(1)

    # Ищем .dem, .dem.gz, .dem.zst
    demos = sorted(
        glob.glob(str(demos_dir / "*.dem")) +
        glob.glob(str(demos_dir / "*.dem.gz")) +
        glob.glob(str(demos_dir / "*.dem.zst"))
    )

    if not demos:
        print(f"[!] В папке {demos_dir} нет .dem файлов")
        sys.exit(1)

    total_size = sum(os.path.getsize(d) for d in demos)
    print("=" * 60)
    print(f"  Найдено {len(demos)} демок в {demos_dir}")
    print(f"  Общий размер: {total_size / 1024**2:.1f} MB")
    print(f"  d_model={args.d_model}  n_layers={args.n_layers}  n_heads={args.n_heads}")
    print(f"  epochs={args.epochs}  seq_len={args.seq_len}  lr={args.lr}")
    print(f"  seed={args.seed}  target_hz={args.target_hz}")
    print(f"  Output: {args.output}  Keep: {'YES' if args.keep else 'NO'}")
    print("=" * 60)

    script_dir = Path(__file__).parent
    main_py = script_dir / "main.py"

    failed = 0
    for i, demo in enumerate(demos):
        name = Path(demo).name
        size_mb = os.path.getsize(demo) / 1024**2 if os.path.exists(demo) else 0
        print(f"\n[{i + 1}/{len(demos)}] {name} ({size_mb:.1f} MB)")

        cmd = [
            sys.executable, str(main_py),
            "--demo", demo,
            "--train",
            "--d-model", str(args.d_model),
            "--n-layers", str(args.n_layers),
            "--n-heads", str(args.n_heads),
            "--epochs", str(args.epochs),
            "--seq-len", str(args.seq_len),
            "--lr", str(args.lr),
            "--seed", str(args.seed),
            "--target-hz", str(args.target_hz),
            "--output", args.output,
        ]
        if args.cpu:
            cmd.append("--cpu")
        if args.dashboard:
            cmd.append("--dashboard")

        result = subprocess.run(cmd, cwd=script_dir)

        if result.returncode == 0:
            print(f"     OK — обучение завершено")
            if not args.keep:
                try:
                    os.remove(demo)
                    print(f"     Удалён: {name}")
                except OSError as e:
                    print(f"     [!] Не удалось удалить: {e}")
        else:
            print(f"     [!] Ошибка (код {result.returncode}), файл сохранён")
            failed += 1

    model_path = Path(args.output) / "model.pt"
    print(f"\n{'=' * 60}")
    print(f"  ГОТОВО. Обработано: {len(demos) - failed}/{len(demos)} демок")
    if failed:
        print(f"  Ошибок: {failed}")
    if model_path.exists():
        size = model_path.stat().st_size / 1024**2
        print(f"  Финальная модель: {model_path.resolve()} ({size:.1f} MB)")
    else:
        print(f"  [!] Файл модели не найден: {model_path}")
    if not args.keep:
        print(f"  ⚠ Демки удалены. В следующий раз используйте --keep.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
