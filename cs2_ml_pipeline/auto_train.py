"""auto_train.py — Train model on all .dem files from a folder incrementally.

Usage:
    python auto_train.py                        # searches for .dem in current dir
    python auto_train.py --dir d:\\demos        # or specify folder
    python auto_train.py --d-model 512          # custom params
    python auto_train.py --keep                 # don't delete demos after processing
"""

import os, sys, glob, subprocess, argparse
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="Train model on a folder of .dem files (incremental)")
    parser.add_argument("--dir", type=str, default=".",
                        help="folder with .dem files (default: current)")
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--n-heads", type=int, default=DEFAULT_N_HEADS)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seq-len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--target-hz", type=int, default=DEFAULT_TARGET_HZ)
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--keep", action="store_true", help="do NOT delete demos after training")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dashboard", action="store_true")
    args = parser.parse_args()

    demos_dir = Path(args.dir)
    if not demos_dir.exists():
        print(f"[!] Folder not found: {demos_dir}")
        sys.exit(1)

    demos = sorted(
        glob.glob(str(demos_dir / "*.dem")) +
        glob.glob(str(demos_dir / "*.dem.gz")) +
        glob.glob(str(demos_dir / "*.dem.zst"))
    )

    if not demos:
        print(f"[!] No .dem files found in {demos_dir}")
        sys.exit(1)

    total_size = sum(os.path.getsize(d) for d in demos)
    print("=" * 60)
    print(f"  Found {len(demos)} demos in {demos_dir}")
    print(f"  Total size: {total_size / 1024**2:.1f} MB")
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
            print(f"     OK — training complete")
            if not args.keep:
                try:
                    os.remove(demo)
                    print(f"     Deleted: {name}")
                except OSError as e:
                    print(f"     [!] Could not delete: {e}")
        else:
            print(f"     [!] Error (code {result.returncode}), file kept")
            failed += 1

    model_path = Path(args.output) / "model.pt"
    print(f"\n{'=' * 60}")
    print(f"  DONE. Processed: {len(demos) - failed}/{len(demos)} demos")
    if failed:
        print(f"  Errors: {failed}")
    if model_path.exists():
        size = model_path.stat().st_size / 1024**2
        print(f"  Final model: {model_path.resolve()} ({size:.1f} MB)")
    else:
        print(f"  [!] Model file not found: {model_path}")
    if not args.keep:
        print(f"  ⚠ Demos deleted. Next time use --keep.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
