"""main.py — Full CS2 ML Pipeline: .dem → tokens → model → dashboard.

Usage:
    python main.py --demo path/to/match.dem --train --dashboard
    python main.py --demo-mode --train --dashboard
"""

import sys
import os
import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

# Clean imports from the cs2_ml_pipeline package
from cs2_ml_pipeline.etl.parser import CS2DemoParser, DemoParseError
from cs2_ml_pipeline.etl.aligner import TickAligner
from cs2_ml_pipeline.etl.segmenter import RoundSegmenter
from cs2_ml_pipeline.etl.exporter import TickExporter
from cs2_ml_pipeline.etl.validators import DataValidator

from cs2_ml_pipeline.tokenizer.hybrid import HybridTokenizer
from cs2_ml_pipeline.tokenizer.events import EventEmbedder, EVENT_VOCAB, VOCAB_SIZE
from cs2_ml_pipeline.tokenizer.map_layers import MapLayerGenerator

from cs2_ml_pipeline.models.transformer import CS2Transformer
from cs2_ml_pipeline.models.heads import DeathHead, ValueHead
from cs2_ml_pipeline.models.map_ae import MapAutoencoder
from cs2_ml_pipeline.models.map_conditioning import MapTokenInjector

from cs2_ml_pipeline.training.trainer import CS2Trainer
from cs2_ml_pipeline.training.curriculum import CurriculumScheduler
from cs2_ml_pipeline.training.proxy_metrics import ProxyMetricGenerator
from cs2_ml_pipeline.training.contrastive import SkillContrastiveHead

from cs2_ml_pipeline.inference.streamer import StreamingInferenceEngine
from cs2_ml_pipeline.inference.xai import XAIModule
from cs2_ml_pipeline.inference.dashboard import CoachingDashboard


# ── Event type → vocabulary mapping ─────────────────────────────────────
EVENT_TYPE_TO_VOCAB = {
    "round_start": "ROUND_START", "round_end": "ROUND_END",
    "plant": "PLANT_START", "plant_start": "PLANT_START",
    "defuse": "DEFUSE_START", "defuse_start": "DEFUSE_START",
    "explode": "BOMB_EXPLODED",
    "bomb_pickup": "BOMB_PICKUP", "bomb_drop": "BOMB_DROP",
    "death": "DEATH", "kill": "KILL",
    "weapon_fire": "DAMAGE", "item_pickup": "BOMB_PICKUP",
}
ROUND_META_EVENTS = {"ROUND_START", "ROUND_END", "FREEZE_END"}


# ── Pipeline Config ─────────────────────────────────────────────────────
class PipelineConfig:
    d_model: int = 256
    n_layers: int = 8
    n_heads: int = 4
    target_hz: int = 8
    seq_len: int = 128
    batch_size: int = 4
    epochs: int = 5
    steps_per_epoch: int = 100
    lr: float = 3e-4
    weight_decay: float = 0.01
    dropout: float = 0.1
    map_ae_epochs: int = 50
    val_split: float = 0.2
    seed: int = 42
    device: str = "auto"
    output_dir: str = "output"

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)


# ── Synthetic demo generator (fallback when no .dem file) ────────────────
def generate_synthetic_demo():
    """Generate synthetic CS2 demo data for testing without real .dem files."""
    import numpy as np
    import pandas as pd

    TICK_RATE = 64
    PLAYER_NAMES = [
        "CT_Player1", "CT_Player2", "CT_Player3", "CT_Player4", "CT_Player5",
        "T_Player1", "T_Player2", "T_Player3", "T_Player4", "T_Player5",
    ]
    ROUND_CONFIGS = [
        {"freeze_end": 500, "end_official": 3500, "game_end": 3550, "winner": "CT", "reason": "ct_killed"},
        {"freeze_end": 4000, "end_official": 7000, "game_end": 7050, "winner": "T", "reason": "bomb_exploded"},
        {"freeze_end": 7500, "end_official": 10500, "game_end": 10550, "winner": "CT", "reason": "bomb_defused"},
    ]

    class SyntheticDemo:
        def __init__(self):
            self.ticks = pd.DataFrame()
            self.rounds = pd.DataFrame()
            self.kills = pd.DataFrame()
            self.events = {}

        def parse(self):
            rng = np.random.RandomState(42)
            ticks_list = []
            for rnd in ROUND_CONFIGS:
                start, end = rnd["freeze_end"], rnd["game_end"]
                for t in range(start, end + 1, 4):
                    for p in PLAYER_NAMES:
                        team = "CT" if "CT" in p else "T"
                        base_x = -1500 if team == "CT" else 1500
                        alive = 100 if (t - start) < (end - start) * 0.8 or "Player1" in p else 0
                        ticks_list.append({
                            "tick": t, "name": p, "team": team,
                            "x": base_x + rng.randn() * 200,
                            "y": rng.randn() * 200,
                            "z": 64.0 + rng.randn() * 5,
                            "yaw": rng.uniform(0, 360),
                            "health": alive, "armor": 100,
                            "has_helmet": True, "has_defuser": "CT_Player1" in p,
                            "is_alive": alive > 0,
                        })
            self.ticks = pd.DataFrame(ticks_list)

            rounds_list = []
            for i, rnd in enumerate(ROUND_CONFIGS):
                rounds_list.append({
                    "freeze_end": rnd["freeze_end"],
                    "official_end": rnd["end_official"],
                    "game_end": rnd["game_end"],
                    "winner": rnd["winner"],
                    "reason": rnd["reason"],
                })
            self.rounds = pd.DataFrame(rounds_list)

            kills_list = []
            for rnd in ROUND_CONFIGS:
                mid = (rnd["freeze_end"] + rnd["end_official"]) // 2
                kills_list.append({
                    "tick": mid, "attacker_name": "T_Player1",
                    "victim_name": "CT_Player3", "weapon": "ak47",
                })
            self.kills = pd.DataFrame(kills_list)

            self.events = {
                "round_start": pd.DataFrame([{"tick": r["freeze_end"] - 100} for r in ROUND_CONFIGS]),
                "round_end": pd.DataFrame([{"tick": r["end_official"], "winner": r["winner"]} for r in ROUND_CONFIGS]),
            }
            return self

    return SyntheticDemo()


# ── Pipeline ────────────────────────────────────────────────────────────
class CS2Pipeline:
    def __init__(self, config: PipelineConfig = None):
        self.cfg = config or PipelineConfig()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
            if self.cfg.device == "auto" else torch.device(self.cfg.device)
        self.output_dir = Path(self.cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        self.rounds, self.game_events = {}, []
        self.kills_df = pd.DataFrame()
        self.map_name, self.tick_rate = "unknown", 64.0
        self.map_tensor, self.z_map = None, None
        self.tokenizer = self.map_ae = self.map_injector = None
        self.model = self.death_head = self.value_head = None
        self.tokens_train = self.death_t_train = self.value_t_train = None
        self.tokens_val = self.death_t_val = self.value_t_val = None
        self.token_player_names = []
        self.dataset_dir: Path = None

    # ── Stage 1: ETL ────────────────────────────────────────────────────
    def run_etl(self, demo_path=None, use_synthetic=False):
        print(f"\n{'='*50}\n  STAGE 1: ETL\n{'='*50}")

        parse_ok = True
        if use_synthetic or demo_path is None:
            print("[*] Synthetic demo")
            parse_ok = False
        else:
            try:
                import tempfile
                actual_path = demo_path
                temp_dem_path = None
                if demo_path.endswith(".zst"):
                    try:
                        import zstandard
                        tmp = tempfile.NamedTemporaryFile(suffix=".dem", delete=False)
                        with open(demo_path, "rb") as f_in:
                            dctx = zstandard.ZstdDecompressor()
                            dctx.copy_stream(f_in, tmp)
                        tmp.close()
                        actual_path = tmp.name
                        temp_dem_path = actual_path
                        print(f"  Decompressed .zst to temp {actual_path}")
                    except ImportError:
                        print("[!] zstandard not installed: pip install zstandard")
                        parse_ok = False
                elif demo_path.endswith(".gz"):
                    import gzip, shutil
                    tmp = tempfile.NamedTemporaryFile(suffix=".dem", delete=False)
                    with gzip.open(demo_path, "rb") as f_in:
                        shutil.copyfileobj(f_in, tmp)
                    tmp.close()
                    actual_path = tmp.name
                    temp_dem_path = actual_path
                    print(f"  Decompressed .gz to temp {actual_path}")

                print(f"[*] Parsing: {actual_path}")
                parser = CS2DemoParser(actual_path)
                try:
                    parser.parse()
                except (DemoParseError, Exception) as e:
                    print(f"[!] Parse failed: {e}")
                    print("[!] Falling back to synthetic data")
                    parse_ok = False
                finally:
                    if temp_dem_path and os.path.exists(temp_dem_path):
                        try:
                            os.remove(temp_dem_path)
                        except OSError:
                            pass
            except Exception as e:
                print(f"[!] Error: {e}")
                parse_ok = False

        if not parse_ok:
            gen = generate_synthetic_demo()
            gen.parse()
            ticks_df = gen.ticks
            rounds_df = gen.rounds
            self.map_name = "de_mirage"
            self.tick_rate = 64.0
            self.game_events = []
            kills_df = gen.kills if hasattr(gen, "kills") else pd.DataFrame()
        else:
            self.map_name = parser.map_name or "unknown"
            self.tick_rate = parser.tick_rate
            ticks_df = parser.demo.ticks.to_pandas().copy()
            rounds_df = parser.demo.rounds.to_pandas().copy()
            events = parser.demo.events if hasattr(parser.demo, "events") else {}
            kills_df = parser.demo.kills.to_pandas().copy() if hasattr(parser.demo, "kills") else pd.DataFrame()
            self.kills_df = kills_df
            extra_df, c4_df = pd.DataFrame(), pd.DataFrame()
            if parser.extra_parser:
                try:
                    extra_df = parser.extra_parser.parse_ticks(["yaw", "armor", "has_helmet", "has_defuser"])
                    c4_df = parser.extra_parser.parse_ticks(["X", "Y"], ticks_target=["weapon_c4"])
                    if c4_df.empty:
                        c4_df = parser.extra_parser.parse_ticks(["X", "Y"], ticks_target=["C_WeaponC4"])
                except Exception:
                    pass
            aligner = TickAligner(ticks_df, rounds_df, events, kills_df=kills_df)
            if not extra_df.empty:
                aligner.merge_extra(extra_df)
            if not c4_df.empty:
                aligner.merge_c4(c4_df)
            aligned = aligner.get_aligned()
            self.game_events = aligner.extract_game_events()
            segmenter = RoundSegmenter(aligned, rounds_df, self.game_events, self.tick_rate)
            self.rounds = segmenter.segment()
            print(f"  Map: {self.map_name}  Tickrate: {self.tick_rate}")

        exporter = TickExporter(self.rounds, tick_rate=self.tick_rate,
                                target_hz=self.cfg.target_hz,
                                output_dir=str(self.output_dir / "dataset"),
                                map_name=self.map_name)
        result = exporter.export_all()
        self.dataset_dir = self.output_dir / "dataset" / self.map_name
        validator = DataValidator(result["parquet_paths"])
        passed = validator.validate_all()
        print(f"  Rounds: {result['num_rounds']}  Parquet: {len(result['parquet_paths'])} files")
        print(f"  Validation: {'PASSED' if passed else 'FAILED'}")
        return self.rounds

    # ── Map layers & MapAE ───────────────────────────────────────────────
    def generate_map_layers(self):
        gen = MapLayerGenerator(map_name=self.map_name, resolution=256)
        self.map_tensor = torch.from_numpy(gen.generate()).unsqueeze(0).to(self.device)
        return self.map_tensor

    def train_map_ae(self):
        if self.map_tensor is None:
            self.generate_map_layers()
        self.map_ae = MapAutoencoder(in_channels=3, base_channels=32,
                                     z_dim=128, input_size=256, beta=0.1).to(self.device)
        opt = optim.AdamW(self.map_ae.parameters(), lr=1e-3, weight_decay=0.01)
        self.map_ae.train()
        print(f"  Map-AE training ({self.cfg.map_ae_epochs} epochs)...")
        for ep in range(self.cfg.map_ae_epochs):
            opt.zero_grad()
            out = self.map_ae(self.map_tensor)
            loss = self.map_ae.compute_loss(self.map_tensor, out)["total"]
            loss.backward()
            opt.step()
            if (ep + 1) % 10 == 0:
                print(f"    epoch {ep+1:3d}: loss={loss.item():.4f}")
        self.map_ae.eval()
        with torch.no_grad():
            self.z_map = self.map_ae.encode(self.map_tensor)
        print(f"  z_map cached: {self.z_map.shape}")

    # ── Stage 2: Tokenization ────────────────────────────────────────────
    def build_tokenizer(self):
        print(f"\n{'='*50}\n  STAGE 2: Tokenization\n{'='*50}")
        self.tokenizer = HybridTokenizer(d_model=self.cfg.d_model).to(self.device).eval()
        print(f"  Tokenizer params: {sum(p.numel() for p in self.tokenizer.parameters()):,}")
        return self.tokenizer

    def get_players(self):
        seen = set()
        pcol = None
        for rdf in self.rounds.values():
            if "name" in rdf.columns:
                pcol = "name"; break
            if "player_name" in rdf.columns:
                pcol = "player_name"; break
        if pcol:
            for rdf in self.rounds.values():
                for v in rdf[pcol].astype(str):
                    seen.add(v.strip())
        return sorted(p for p in seen if p != "C4_ENTITY")

    def filter_to_player(self, player_name):
        for rnum in list(self.rounds.keys()):
            rdf = self.rounds[rnum]
            pcol = "name" if "name" in rdf.columns else \
                   "player_name" if "player_name" in rdf.columns else None
            if pcol:
                mask = rdf[pcol].astype(str).str.strip() == player_name.strip()
                rdf = rdf.loc[mask].copy()
            if rdf.empty:
                del self.rounds[rnum]
            else:
                self.rounds[rnum] = rdf

    def tokenize(self):
        if self.tokenizer is None:
            self.build_tokenizer()
        if self.map_tensor is None:
            self.generate_map_layers()
        print(f"\n  Tokenizing {len(self.rounds)} rounds...")

        all_tokens, all_death, all_value = [], [], []
        for rnum in sorted(self.rounds.keys()):
            rdf = self.rounds[rnum].copy()
            if rdf.empty:
                continue
            step = max(1, int(self.tick_rate / self.cfg.target_hz))
            rdf = rdf.sort_values("tick").reset_index(drop=True)
            rdf = rdf.iloc[::step].copy()
            abs_first_tick = rdf["tick"].min()
            self._normalize_round(rdf)
            self._attach_event_ids(rdf, abs_first_tick)
            self._attach_cover_scores(rdf)
            for c in ["dx_to_c4", "dy_to_c4", "yaw_cos", "yaw_sin", "state_mask"]:
                if c not in rdf.columns:
                    rdf[c] = 0.0
            n = len(rdf)
            pos = torch.zeros(1, n, 5, dtype=torch.float32)
            for col, idx in [("dx_to_c4", 0), ("dy_to_c4", 1)]:
                if col in rdf.columns:
                    pos[0, :, idx] = torch.tensor(rdf[col].fillna(0).astype(float).values, dtype=torch.float32)
            if "z" in rdf.columns:
                pos[0, :, 2] = torch.tensor(rdf["z"].fillna(0).astype(float).values, dtype=torch.float32)
            orient = torch.zeros(1, n, 3, dtype=torch.float32)
            for col, idx in [("yaw_cos", 0), ("yaw_sin", 1)]:
                if col in rdf.columns:
                    orient[0, :, idx] = torch.tensor(rdf[col].fillna(0).astype(float).values, dtype=torch.float32)
            state = torch.zeros(1, n, 3, dtype=torch.float32)
            for col, idx in [("health", 0), ("armor", 1)]:
                if col in rdf.columns:
                    vals = np.clip(rdf[col].fillna(0).astype(float).values / 100.0, 0, 1)
                    state[0, :, idx] = torch.tensor(vals, dtype=torch.float32)
            cover = torch.zeros(1, n, 1, dtype=torch.float32)
            if "cover_score" in rdf.columns:
                cover[0, :, 0] = torch.tensor(rdf["cover_score"].fillna(0).astype(float).values, dtype=torch.float32)
            events = torch.zeros(1, n, dtype=torch.long)
            if "event_id" in rdf.columns:
                events[0, :] = torch.tensor(rdf["event_id"].fillna(0).astype(int).values, dtype=torch.long)
            batch = {"pos": pos, "orient": orient, "state": state, "cover": cover, "events": events}
            with torch.no_grad():
                tokens = self.tokenizer.forward_dict({k: v.to(self.device) for k, v in batch.items()})
            death_t = torch.zeros(1, n, 1, dtype=torch.float32)
            if "state_mask" in rdf.columns:
                mask_vals = rdf["state_mask"].fillna(1).astype(int).values
                for t in range(1, n):
                    if mask_vals[t - 1] == 1 and mask_vals[t] == 0:
                        death_t[0, t, 0] = 1.0
            value_t = self._compute_value_targets(rdf, n, abs_first_tick)
            pcol = "name" if "name" in rdf.columns else \
                   "player_name" if "player_name" in rdf.columns else None
            if pcol:
                self.token_player_names.extend(rdf[pcol].astype(str).values.tolist())
            else:
                self.token_player_names.extend(["Player"] * n)
            all_tokens.append(tokens)
            all_death.append(death_t)
            all_value.append(value_t)

        if not all_tokens:
            print("[!] No rounds, using synthetic")
            return self._tokenize_synthetic()

        tokens_cat = torch.cat(all_tokens, dim=1)
        death_cat = torch.cat(all_death, dim=1)
        value_cat = torch.cat(all_value, dim=1)
        total = tokens_cat.shape[1]
        val_split = min(int(total * self.cfg.val_split), max(1, total // 4)) if total >= 8 else 0
        val_start = total - val_split
        self.tokens_train = tokens_cat[:, :val_start, :] if val_start > 0 else tokens_cat
        self.death_t_train = death_cat[:, :val_start, :] if val_start > 0 else death_cat
        self.value_t_train = value_cat[:, :val_start, :] if val_start > 0 else value_cat
        if val_split > 0:
            self.tokens_val = tokens_cat[:, val_start:, :]
            self.death_t_val = death_cat[:, val_start:, :]
            self.value_t_val = value_cat[:, val_start:, :]
        print(f"  Tokens: {self.tokens_train.shape} train + {val_split} val  "
              f"deaths: {self.death_t_train.sum().item():.0f} train / "
              f"{self.death_t_val.sum().item() if self.tokens_val is not None else 0:.0f} val")
        return self.tokens_train, self.death_t_train, self.value_t_train

    def _normalize_round(self, df):
        if "yaw" in df.columns:
            yaw = np.radians(df["yaw"].fillna(0).astype(float))
            df["yaw_cos"] = np.cos(yaw)
            df["yaw_sin"] = np.sin(yaw)
        else:
            df["yaw_cos"], df["yaw_sin"] = 1.0, 0.0
        if "player_name" in df.columns and "x" in df.columns:
            c4_rows = df[df["player_name"] == "C4_ENTITY"]
            if not c4_rows.empty and not pd.isna(c4_rows["x"].iloc[0]):
                cx, cy = float(c4_rows["x"].iloc[0]), float(c4_rows["y"].iloc[0])
            else:
                cx = df["x"].mean() if not df["x"].isna().all() else 0.0
                cy = df["y"].mean() if not df["y"].isna().all() else 0.0
            df["dx_to_c4"] = df["x"].astype(float) - cx
            df["dy_to_c4"] = df["y"].astype(float) - cy
        df["state_mask"] = (df["health"] > 0).astype(int) if "health" in df.columns else 1

    def _attach_event_ids(self, df, abs_first_tick=0):
        df["event_id"] = 0
        if not self.game_events:
            return
        ev_by_tick = {}
        for ev in self.game_events:
            key = (ev["tick"], ev.get("player", "Player"))
            ev_type = EVENT_TYPE_TO_VOCAB.get(ev["type"])
            if ev_type and ev_type not in ROUND_META_EVENTS:
                ev_by_tick.setdefault(key, []).append(ev_type)
        for idx, row in df.iterrows():
            t = int(row["tick"]) + int(abs_first_tick)
            p = str(row.get("name", "Player"))
            evts = ev_by_tick.get((t, p), []) + ev_by_tick.get((t, "Player"), [])
            if evts:
                df.at[df.index[idx], "event_id"] = EventEmbedder.resolve_event_id(evts)

    def _attach_cover_scores(self, df):
        df["cover_score"] = 0.0
        if "x" not in df.columns:
            return
        cover = self.map_tensor[0, 2].cpu().numpy()
        H, W = cover.shape
        xv = df["x"].fillna(0).astype(float).values
        yv = df["y"].fillna(0).astype(float).values
        scale = 5000.0
        xi = np.clip(((xv + scale) / (2 * scale) * (W - 1)), 0, W - 1).astype(int)
        yi = np.clip(((scale - yv) / (2 * scale) * (H - 1)), 0, H - 1).astype(int)
        df["cover_score"] = cover[yi, xi]

    def _compute_value_targets(self, df, n, abs_first_tick=0):
        """Compute simple value targets from round victory."""
        winner = df.attrs.get("winner", "unknown") if hasattr(df, "attrs") else "unknown"
        value_t = torch.zeros(1, n, 1, dtype=torch.float32)
        if winner != "unknown":
            last_third = max(1, n // 3)
            sign = 1.0 if winner == "CT" else -1.0
            value_t[0, -last_third:, 0] = sign * 0.5
        return value_t

    def _tokenize_synthetic(self):
        """Generate synthetic tokens when no real data available."""
        B, S = self.cfg.batch_size, self.cfg.seq_len
        self.tokens_train = torch.randn(B, S, self.cfg.d_model).to(self.device)
        self.death_t_train = torch.zeros(B, S, 1).to(self.device)
        self.death_t_train[torch.rand(B, S, 1) > 0.95] = 1.0
        self.value_t_train = (torch.rand(B, S, 1) * 2 - 1).to(self.device)
        self.tokens_val = self.tokens_train[:, :max(1, S // 4), :]
        self.death_t_val = self.death_t_train[:, :max(1, S // 4), :]
        self.value_t_val = self.value_t_train[:, :max(1, S // 4), :]
        return self.tokens_train, self.death_t_train, self.value_t_train

    # ── Stage 3: Build model ────────────────────────────────────────────
    def build_model(self):
        print(f"\n{'='*50}\n  STAGE 3: Transformer + Heads + MapAE\n{'='*50}")
        if self.map_ae is None:
            self.train_map_ae()
        self.map_injector = MapTokenInjector(z_dim=128, d_model=self.cfg.d_model).to(self.device)
        self.model = CS2Transformer(d_model=self.cfg.d_model, n_layers=self.cfg.n_layers,
                                    n_heads=self.cfg.n_heads, dropout=self.cfg.dropout).to(self.device)
        self.death_head = DeathHead(d_model=self.cfg.d_model, hidden_dim=self.cfg.d_model // 4).to(self.device)
        self.value_head = ValueHead(d_model=self.cfg.d_model, hidden_dim=self.cfg.d_model // 4).to(self.device)
        self.contrastive_head = SkillContrastiveHead(d_model=self.cfg.d_model, proj_dim=128).to(self.device)
        self.model.death_head = self.death_head
        self.model.value_head = self.value_head
        self.model.contrastive_head = self.contrastive_head
        print(f"  Transformer params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  MapAE params: {sum(p.numel() for p in self.map_ae.parameters()):,}")
        return self.model

    # ── Stage 4: Training ───────────────────────────────────────────────
    def train(self):
        print(f"\n{'='*50}\n  STAGE 4: Training\n{'='*50}")
        if self.model is None:
            self.build_model()
        if self.tokens_train is None:
            self.tokenize()

        ckpt_path = self.output_dir / "model.pt"
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location=self.device)
                if ckpt.get("config", {}).get("d_model") == self.cfg.d_model:
                    self.model.load_state_dict(ckpt["model"])
                    self.death_head.load_state_dict(ckpt["death_head"])
                    self.value_head.load_state_dict(ckpt["value_head"])
                    print(f"[*] Loaded checkpoint from {ckpt_path}")
            except RuntimeError as e:
                print(f"[!] Checkpoint load failed: {e}")

        self.tokens_train = self.tokens_train.to(self.device)
        self.death_t_train = self.death_t_train.to(self.device)
        self.value_t_train = self.value_t_train.to(self.device)
        if self.tokens_val is not None:
            self.tokens_val = self.tokens_val.to(self.device)
            self.death_t_val = self.death_t_val.to(self.device)
            self.value_t_val = self.value_t_val.to(self.device)

        with torch.no_grad():
            tokens_wm = self.map_injector(self.tokens_train, self.z_map)
        B = self.tokens_train.shape[0]
        pad_d = torch.zeros(B, 1, 1, device=self.device)
        death_t = torch.cat([pad_d, self.death_t_train], dim=1)
        value_t = torch.cat([pad_d, self.value_t_train], dim=1)
        tokens_val_wm = death_val = value_val = None
        if self.tokens_val is not None:
            with torch.no_grad():
                tokens_val_wm = self.map_injector(self.tokens_val, self.z_map)
            Bv = self.tokens_val.shape[0]
            pad_dv = torch.zeros(Bv, 1, 1, device=self.device)
            death_val = torch.cat([pad_dv, self.death_t_val], dim=1)
            value_val = torch.cat([pad_dv, self.value_t_val], dim=1)

        opt = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        trainer = CS2Trainer(self.model, opt, device=self.device, clip_grad_norm=1.0,
                             contrastive_head=self.contrastive_head)
        sched = CurriculumScheduler(total_steps=self.cfg.epochs * self.cfg.steps_per_epoch,
                                    seq_len_start=min(64, self.cfg.seq_len // 2),
                                    seq_len_end=self.cfg.seq_len)
        best_val = float("inf")
        T = tokens_wm.shape[1]
        for ep in range(self.cfg.epochs):
            self.model.train()
            el = {"ntp": 0.0, "death": 0.0, "value": 0.0, "contrastive": 0.0}
            for step_i in range(self.cfg.steps_per_epoch):
                sched.step()
                sl = min(sched.get_seq_len(), T)
                start = torch.randint(0, T - sl + 1, (1,)).item() if T > sl else 0
                weights = sched.get_loss_weights()
                r = trainer.multi_task_step(
                    tokens_wm[:, start:start + sl, :],
                    death_t[:, start:start + sl, :],
                    value_t[:, start:start + sl, :],
                    loss_weights=weights,
                )
                for k in ["ntp", "death", "value"]:
                    el[k] += r.get(k, 0.0)
                if weights.get("contrastive", 0) > 0 and step_i % 2 == 0:
                    rc = trainer.contrastive_step(
                        tokens_wm[:, start:start + sl, :],
                        value_t[:, start:start + sl, :],
                        window_size=min(16, sl // 4),
                        contrastive_weight=weights.get("contrastive", 0.1),
                    )
                    el["contrastive"] += rc.get("contrastive", 0.0)

            for k in el:
                el[k] /= self.cfg.steps_per_epoch

            val_str = ""
            if tokens_val_wm is not None:
                self.model.eval()
                with torch.no_grad():
                    v_input = tokens_val_wm[:, :self.cfg.seq_len + 1, :]
                    v_input = v_input[:, :-1, :]
                    v_target = tokens_val_wm[:, 1:self.cfg.seq_len + 1, :]
                    v_out = self.model(v_input)
                    v_hidden = v_out["hidden_states"]
                    v_pred_n = torch.nn.functional.normalize(v_hidden, dim=-1)
                    v_target_n = torch.nn.functional.normalize(v_target, dim=-1)
                    v_ntp = (1.0 - (v_pred_n * v_target_n).sum(dim=-1).mean()).item()
                    val_str = f"  val_ntp={v_ntp:.4f}"
                    if v_ntp < best_val:
                        best_val = v_ntp

            print(f"  Epoch {ep+1:3d}: ntp={el['ntp']:.4f} death={el['death']:.4f} "
                  f"value={el['value']:.4f} contrastive={el['contrastive']:.4f}{val_str}")

        # Save checkpoint
        torch.save({
            "model": self.model.state_dict(),
            "death_head": self.death_head.state_dict(),
            "value_head": self.value_head.state_dict(),
            "map_ae": self.map_ae.state_dict(),
            "config": {"d_model": self.cfg.d_model, "n_layers": self.cfg.n_layers,
                       "n_heads": self.cfg.n_heads, "z_dim": 128},
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    # ── Stage 5: Inference / Dashboard ──────────────────────────────────
    def run_dashboard(self, num_ticks=32, player_name=None, player_num=None):
        print(f"\n{'='*50}\n  STAGE 5: Dashboard\n{'='*50}")
        if self.model is None:
            self.build_model()
        if self.tokens_train is None:
            self.tokenize()
        if self.z_map is None:
            self.train_map_ae()

        dash = CoachingDashboard()
        streamer = StreamingInferenceEngine(self.model, window_size=256, device=self.device)
        xai_mod = XAIModule(self.model)

        tokens = self.tokens_train[:, :min(num_ticks, self.tokens_train.shape[1]), :]
        with torch.no_grad():
            tokens_inj = self.map_injector(tokens, self.z_map)

        streamer.reset()
        with torch.no_grad():
            out = self.model(tokens_inj)
            hidden = out["hidden_states"]
            death_preds = self.death_head(hidden)
            value_preds = self.value_head(hidden)

        # Skip the [MAP] token for display
        for t in range(min(num_ticks, hidden.shape[1])):
            token_idx = t
            dp = float(death_preds[0, token_idx, 0].item())
            dp = max(0.0, min(1.0, dp))
            vt = value_preds[0, token_idx, 0].item()
            val_death = max(-1.0, min(1.0, 1.0 - dp * 2))
            val_event = min(1.0, max(-1.0, vt))
            val = val_death + val_event * 0.5
            val = max(-1.0, min(1.0, val))
            score = max(0.0, min(1.0, 1.0 - dp * 3))
            hint = XAIModule.generate_hint(score, dp, self.map_name)
            payload = dash.process_tick(tick=t, score=score, death_prob=dp, value=val, hint=hint)
            bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
            score_str = f"[{bar}] {score:.2f}"
            print(f"  {t:4d} | {payload.color:>5s} | {score_str:<28s} | {dp:5.2f} | {val:+5.2f} | {payload.hint}")

        log = self.output_dir / "dashboard_log.json"
        log.write_text(json.dumps(dash.get_history(), indent=2), encoding="utf-8")
        print(f"\n  Dashboard log: {log.resolve()}")


# ── CLI ───────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="CS2 End-to-End Pipeline")
    p.add_argument("--demo", type=str, help="Path to .dem file")
    p.add_argument("--demo-mode", action="store_true", help="Use synthetic data")
    p.add_argument("--output", type=str, default="output")
    p.add_argument("--train", action="store_true")
    p.add_argument("--dashboard", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--target-hz", type=int, default=8)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ticks", type=int, default=32)
    p.add_argument("--player", type=str, default=None)
    p.add_argument("--player-num", type=int, default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    cfg = PipelineConfig(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
                          target_hz=args.target_hz, seq_len=args.seq_len, epochs=args.epochs,
                          lr=args.lr, seed=args.seed, device="cpu" if args.cpu else "auto",
                          output_dir=args.output)
    pipe = CS2Pipeline(config=cfg)
    pipe.run_etl(demo_path=args.demo, use_synthetic=args.demo_mode or args.demo is None)
    if not args.train and not args.dashboard:
        print("\n[DONE] ETL only. Use --train --dashboard for full pipeline.")
        return

    players = pipe.get_players()
    if players:
        print(f"\n  Players ({len(players)}):")
        for i, p in enumerate(players):
            print(f"    [{i+1}] {p}")

    if args.player_num is not None:
        pname = None
        if 1 <= args.player_num <= len(players):
            pname = players[args.player_num - 1]
        if pname:
            pipe.filter_to_player(pname)
            print(f"  -> Filtered to player: {pname}")
    elif args.player:
        pipe.filter_to_player(args.player)
        print(f"  -> Filtered to player: {args.player}")
    elif players:
        try:
            choice = input(f"\n  Select player [1-{len(players)}] or Enter for all: ").strip()
            if choice:
                num = int(choice)
                if 1 <= num <= len(players):
                    pname = players[num - 1]
                    pipe.filter_to_player(pname)
                    print(f"  -> Filtered to player: {pname}")
        except (ValueError, EOFError):
            pass

    pipe.build_tokenizer()
    pipe.tokenize()
    pipe.build_model()
    if args.train:
        pipe.train()
    if args.dashboard:
        pipe.run_dashboard(num_ticks=args.ticks)
    print(f"\n{'='*50}\n  PIPELINE COMPLETE\n  Output: {Path(args.output).resolve()}\n{'='*50}")


if __name__ == "__main__":
    main()
