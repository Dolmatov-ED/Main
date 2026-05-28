"""v6_end_to_end/main.py — Full CS2 ML Pipeline: .dem → tokens → model → dashboard.
Fixed integration: events, cover, MapAE training, proxy metrics, validation.
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

ROOT = Path(__file__).resolve().parent.parent

# ── Dynamic imports from v1–v5 ──────────────────────────────────────────
def _clear():
    for k in list(sys.modules.keys()):
        if "cs2_ml_pipeline" in k:
            del sys.modules[k]

def _import(ver, imp):
    for v in list(sys.path):
        if any(x in v for x in ["v1_etl", "v2_token", "v3_arch", "v4_pre", "v5_inf"]):
            sys.path.remove(v)
    _clear()
    sys.path.insert(0, str(ROOT / ver))
    ns = {}
    exec(f"from {imp}", ns)
    return ns

_mod = {}
_mod.update(_import("v1_etl_pipeline", "cs2_ml_pipeline.etl.aligner import TickAligner"))
_mod.update(_import("v1_etl_pipeline", "cs2_ml_pipeline.etl.segmenter import RoundSegmenter"))
_mod.update(_import("v1_etl_pipeline", "cs2_ml_pipeline.etl.exporter import TickExporter"))
_mod.update(_import("v1_etl_pipeline", "cs2_ml_pipeline.etl.validators import DataValidator"))
_mod.update(_import("v1_etl_pipeline", "cs2_ml_pipeline.mocks.mock_demo import generate_synthetic_demo"))
_mod.update(_import("v1_etl_pipeline", "cs2_ml_pipeline.etl.parser import CS2DemoParser, DemoParseError"))
_mod.update(_import("v2_tokenization", "cs2_ml_pipeline.tokenizer.hybrid import HybridTokenizer"))
_mod.update(_import("v2_tokenization", "cs2_ml_pipeline.tokenizer.events import EventEmbedder, EVENT_VOCAB, VOCAB_SIZE"))
_mod.update(_import("v2_tokenization", "cs2_ml_pipeline.tokenizer.map_layers import MapLayerGenerator"))
_mod.update(_import("v2_tokenization", "cs2_ml_pipeline.mocks.mock_tokenizer import BatchGenerator"))
_mod.update(_import("v3_architecture", "cs2_ml_pipeline.models.transformer import CS2Transformer"))
_mod.update(_import("v3_architecture", "cs2_ml_pipeline.models.heads import DeathHead, ValueHead"))
_mod.update(_import("v3_architecture", "cs2_ml_pipeline.models.map_ae import MapAutoencoder"))
_mod.update(_import("v3_architecture", "cs2_ml_pipeline.models.map_conditioning import MapTokenInjector"))
_mod.update(_import("v4_pretraining", "cs2_ml_pipeline.training.trainer import CS2Trainer"))
_mod.update(_import("v4_pretraining", "cs2_ml_pipeline.training.curriculum import CurriculumScheduler"))
_mod.update(_import("v4_pretraining", "cs2_ml_pipeline.training.proxy_metrics import ProxyMetricGenerator"))
_mod.update(_import("v4_pretraining", "cs2_ml_pipeline.training.contrastive import SkillContrastiveHead"))
_mod.update(_import("v4_pretraining", "cs2_ml_pipeline.mocks.mock_training import make_death_targets, make_value_targets"))
_mod.update(_import("v5_inference", "cs2_ml_pipeline.inference.streamer import StreamingInferenceEngine"))
_mod.update(_import("v5_inference", "cs2_ml_pipeline.inference.xai import XAIModule"))
_mod.update(_import("v5_inference", "cs2_ml_pipeline.inference.dashboard import CoachingDashboard"))
g = globals()
[g.setdefault(k, v) for k, v in _mod.items()]

# Fake package for trainer runtime imports
import types
_fake_models = types.ModuleType("cs2_ml_pipeline.models")
_fake_models.heads = types.ModuleType("cs2_ml_pipeline.models.heads")
_fake_models.heads.DeathHead = DeathHead
_fake_models.heads.ValueHead = ValueHead
_fake_training = types.ModuleType("cs2_ml_pipeline.training")
_fake_contrastive = types.ModuleType("cs2_ml_pipeline.training.contrastive")
_fake_pkg = types.ModuleType("cs2_ml_pipeline")
_fake_pkg.models = _fake_models
_fake_pkg.training = _fake_training
_fake_training.contrastive = _fake_contrastive
sys.modules["cs2_ml_pipeline"] = _fake_pkg
sys.modules["cs2_ml_pipeline.models"] = _fake_models
sys.modules["cs2_ml_pipeline.models.heads"] = _fake_models.heads
sys.modules["cs2_ml_pipeline.training"] = _fake_training
sys.modules["cs2_ml_pipeline.training.contrastive"] = _fake_contrastive

# Load real contrastive module into the fake package
import importlib.util
_contrastive_spec = importlib.util.spec_from_file_location(
    "cs2_ml_pipeline.training.contrastive",
    ROOT / "v4_pretraining" / "cs2_ml_pipeline" / "training" / "contrastive.py"
)
_contrastive_mod = importlib.util.module_from_spec(_contrastive_spec)
sys.modules["cs2_ml_pipeline.training.contrastive"] = _contrastive_mod
_contrastive_spec.loader.exec_module(_contrastive_mod)
# Copy attributes to the fake module
for attr in dir(_contrastive_mod):
    if not attr.startswith("_"):
        setattr(_fake_contrastive, attr, getattr(_contrastive_mod, attr))


# ── Event type → vocabulary mapping ─────────────────────────────────────
EVENT_TYPE_TO_VOCAB = {
    "round_start": "ROUND_START",
    "round_end": "ROUND_END",
    "plant": "PLANT_START",
    "plant_start": "PLANT_START",
    "defuse": "DEFUSE_START",
    "defuse_start": "DEFUSE_START",
    "explode": "BOMB_EXPLODED",
    "bomb_pickup": "BOMB_PICKUP",
    "bomb_drop": "BOMB_DROP",
    "death": "DEATH",
    "kill": "KILL",
    "weapon_fire": "DAMAGE",
    "item_pickup": "BOMB_PICKUP",
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
        self.token_player_names = []  # player_name per token position
        self.dataset_dir: Path = None

    # ── Stage 1 ──────────────────────────────────────────────────────────
    def run_etl(self, demo_path=None, use_synthetic=False):
        print(f"\n{'='*50}\n  STAGE 1: ETL\n{'='*50}")

        parse_ok = True
        temp_dem_path = None  # Track temp file for cleanup
        if use_synthetic or demo_path is None:
            print("[*] Synthetic demo")
            parse_ok = False
        else:
            try:
                # Decompress .zst / .gz files before parsing
                import_shutil = __import__("shutil")
                import_tempfile = __import__("tempfile")
                actual_path = demo_path
                if demo_path.endswith(".zst"):
                    try:
                        import_zst = __import__("zstandard")
                        tmp = import_tempfile.NamedTemporaryFile(suffix=".dem", delete=False)
                        with open(demo_path, "rb") as f_in:
                            dctx = import_zst.ZstdDecompressor()
                            dctx.copy_stream(f_in, tmp)
                        tmp.close()
                        actual_path = tmp.name
                        temp_dem_path = actual_path
                        print(f"  Decompressed .zst to temp {actual_path}")
                    except ImportError:
                        print("[!] zstandard not installed: pip install zstandard")
                        parse_ok = False
                        actual_path = demo_path
                elif demo_path.endswith(".gz"):
                    try:
                        import_gzip = __import__("gzip")
                        tmp = import_tempfile.NamedTemporaryFile(suffix=".dem", delete=False)
                        with import_gzip.open(demo_path, "rb") as f_in:
                            import_shutil.copyfileobj(f_in, tmp)
                        tmp.close()
                        actual_path = tmp.name
                        temp_dem_path = actual_path
                        print(f"  Decompressed .gz to temp {actual_path}")
                    except Exception:
                        pass

                print(f"[*] Parsing: {actual_path}")
                parser = CS2DemoParser(actual_path)
                try:
                    parser.parse()
                except DemoParseError as e:
                    print(f"[!] Parse failed: {e}")
                    print("[!] Falling back to synthetic data")
                    parse_ok = False
                except Exception as e:
                    print(f"[!] Unexpected parse error: {e}")
                    parse_ok = False
            finally:
                # Clean up temp file immediately after parsing
                if temp_dem_path and os.path.exists(temp_dem_path):
                    try:
                        os.remove(temp_dem_path)
                    except OSError:
                        pass

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

    # ── Stage 2 ──────────────────────────────────────────────────────────
    def build_tokenizer(self):
        print(f"\n{'='*50}\n  STAGE 2: Tokenization\n{'='*50}")
        self.tokenizer = HybridTokenizer(d_model=self.cfg.d_model).to(self.device).eval()
        print(f"  Tokenizer params: {sum(p.numel() for p in self.tokenizer.parameters()):,}")
        return self.tokenizer

    def get_players(self):
        """Return list of unique player names from rounds data."""
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
        """Filter self.rounds to keep only data for a specific player."""
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
            abs_first_tick = rdf["tick"].min()  # save before normalization
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
            # Track player names for per-player dashboard filtering
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
        """Value based on future events: kills, deaths, bomb actions in next 3 sec."""
        window_ticks = max(1, min(int(self.tick_rate * 3.0 / self.cfg.target_hz), n // 2))
        value = torch.zeros(1, n, 1)
        if not self.game_events or "tick" not in df.columns:
            return value.to(self.device)
        pcol = "name" if "name" in df.columns else \
               "player_name" if "player_name" in df.columns else None
        if not pcol:
            return value.to(self.device)
        ticks_arr = df["tick"].astype(int).values
        player_arr = df[pcol].astype(str).values
        event_scores = {"death": -1.0, "plant": 0.5,
                        "defuse": 0.7, "explode": -0.5}
        pevents = {}
        for ev in self.game_events:
            t = ev.get("type", "")
            if t not in event_scores:
                continue
            pn = ev.get("player", "Player")
            pevents.setdefault(pn, []).append((ev.get("tick", 0), event_scores[t]))
        step = int(self.tick_rate / self.cfg.target_hz)
        for idx in range(n):
            pn = str(player_arr[idx])
            ct = int(ticks_arr[idx]) + int(abs_first_tick)  # convert to absolute
            total = 0.0
            for etick, escore in pevents.get(pn, []):
                if ct < etick and (etick - ct) <= window_ticks * step:
                    total += escore
            value[0, idx, 0] = max(-1.0, min(1.0, total))
        return value.to(self.device)

    def _kast_from_kills_df(self, df, round_num):
        """Compute per-tick kill and assist masks from kills_df for a specific round."""
        kills_1d = torch.zeros(len(df))
        assists_1d = torch.zeros(len(df))
        if self.kills_df.empty:
            return kills_1d, assists_1d

        # Filter kills in this round
        round_kills = self.kills_df[
            (self.kills_df["tick"] >= df["tick"].min()) &
            (self.kills_df["tick"] <= df["tick"].max())
        ]
        if round_kills.empty:
            return kills_1d, assists_1d

        # Map to nearest tick
        ticks_list = df["tick"].astype(int).values
        player_list = df["player_name"].astype(str).values if "player_name" in df.columns else []
        for _, krow in round_kills.iterrows():
            kt = int(krow["tick"])
            attacker = str(krow.get("attacker_name", krow.get("attacker", "")))
            assister = str(krow.get("assister_name", krow.get("assister", "")))
            # Find closest tick
            idx = (np.abs(ticks_list - kt)).argmin()
            if attacker and player_list and idx < len(player_list):
                # Mark all players in the attacker's team at this tick as "kill"
                # Simplified: mark the closest tick for the attacker
                for pi, pn in enumerate(player_list):
                    at = max(0, pi - 2)
                    bt = min(len(player_list) - 1, pi + 2)
                    if pn == attacker:
                        kills_1d[pi] = 1.0
                    if assister and pn == assister:
                        assists_1d[pi] = 1.0

        return kills_1d, assists_1d

    def _tokenize_synthetic(self):
        gen = BatchGenerator(batch_size=self.cfg.batch_size, seq_len=self.cfg.seq_len, d_model=self.cfg.d_model)
        batch = gen.generate(batch_size=self.cfg.batch_size, seq_len=self.cfg.seq_len)
        with torch.no_grad():
            tokens = self.tokenizer.forward_dict({k: v.to(self.device) for k, v in batch.items()})
        self.tokens_train = tokens
        self.death_t_train = make_death_targets(self.cfg.batch_size, self.cfg.seq_len).to(self.device)
        self.value_t_train = make_value_targets(self.cfg.batch_size, self.cfg.seq_len).to(self.device)
        self.tokens_val = self.tokens_train[:, :max(1, self.cfg.seq_len // 4), :]
        self.death_t_val = self.death_t_train[:, :max(1, self.cfg.seq_len // 4), :]
        self.value_t_val = self.value_t_train[:, :max(1, self.cfg.seq_len // 4), :]
        return self.tokens_train, self.death_t_train, self.value_t_train

    # ── Stage 3 ──────────────────────────────────────────────────────────
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
        print(f"  Contrastive head: {sum(p.numel() for p in self.contrastive_head.parameters()):,}")
        return self.model

    # ── Stage 4 ──────────────────────────────────────────────────────────
    def train(self):
        print(f"\n{'='*50}\n  STAGE 4: Training\n{'='*50}")
        if self.model is None:
            self.build_model()
        if self.tokens_train is None:
            self.tokenize()

        ckpt_path = self.output_dir / "model.pt"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device)
            if ckpt.get("config", {}).get("d_model") == self.cfg.d_model:
                try:
                    self.model.load_state_dict(ckpt["model"])
                    self.death_head.load_state_dict(ckpt["death_head"])
                    self.value_head.load_state_dict(ckpt["value_head"])
                    print(f"[*] Loaded checkpoint from {ckpt_path}")
                except RuntimeError as e:
                    print(f"[!] Checkpoint load failed (dim mismatch?): {e}")
                    print(f"[*] Starting with fresh weights")

        # Ensure all tensors are on the correct device
        self.tokens_train = self.tokens_train.to(self.device)
        self.death_t_train = self.death_t_train.to(self.device)
        self.value_t_train = self.value_t_train.to(self.device)
        if self.tokens_val is not None:
            self.tokens_val = self.tokens_val.to(self.device)
            self.death_t_val = self.death_t_val.to(self.device)
            self.value_t_val = self.value_t_val.to(self.device)

        # Map injection with no_grad: map_injector params are not trained
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

                # Contrastive step: every other step, when weight > 0
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
                val_str = f" val_ntp={v_ntp:.4f}"
                if v_ntp < best_val:
                    best_val = v_ntp
                    self._save_checkpoint(ckpt_path, "best")
            print(f"  Epoch {ep+1}/{self.cfg.epochs}: ntp={el['ntp']:.4f} "
                  f"death={el['death']:.4f} value={el['value']:.4f} "
                  f"contr={el.get('contrastive', 0):.4f}{val_str}")
        self._save_checkpoint(ckpt_path, "final")
        print(f"  Best val_ntp: {best_val:.4f}")

    def _save_checkpoint(self, path, tag="checkpoint"):
        torch.save({
            "model": self.model.state_dict(),
            "death_head": self.death_head.state_dict(),
            "value_head": self.value_head.state_dict(),
            "contrastive_head": self.contrastive_head.state_dict() if self.contrastive_head else {},
            "config": {"d_model": self.cfg.d_model, "n_layers": self.cfg.n_layers,
                       "n_heads": self.cfg.n_heads, "target_hz": self.cfg.target_hz,
                       "map_name": self.map_name, "tag": tag},
        }, path)
        print(f"  Saved {tag}: {path.resolve()}")

    # ── Stage 5 ──────────────────────────────────────────────────────────
    def run_dashboard(self, num_ticks=32, player_name=None, player_num=None):
        print(f"\n{'='*50}\n  STAGE 5: Streaming + Dashboard\n{'='*50}")
        if self.model is None:
            print("[!] No model. Run train() first.")
            return
        ckpt_path = self.output_dir / "model.pt"
        loaded = False
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device)
            cfg_ck = ckpt.get("config", {})
            if cfg_ck.get("d_model") == self.cfg.d_model:
                try:
                    self.model.load_state_dict(ckpt["model"], strict=False)
                    self.death_head.load_state_dict(ckpt["death_head"])
                    self.value_head.load_state_dict(ckpt["value_head"])
                    loaded = True
                    print(f"[*] Loaded checkpoint: map={cfg_ck.get('map_name')}, tag={cfg_ck.get('tag')}")
                except RuntimeError as e:
                    print(f"[!] Checkpoint mismatch: {e}")
        if not loaded:
            print("[!] Running with UNTRAINED weights — outputs will be static")
        self.model.eval()
        tokens = getattr(self, "tokens_train", None)
        if tokens is not None and tokens.shape[1] >= num_ticks + 8:
            T = min(num_ticks, tokens.shape[1])
            tokens_use = tokens[:, :T, :]
        else:
            T = num_ticks
            tokens_use = torch.randn(1, T, self.cfg.d_model, device=self.device)
        if self.z_map is not None:
            tokens_use = self.map_injector(tokens_use, self.z_map)
        streamer = StreamingInferenceEngine(self.model, window_size=256, device=self.device)
        dash = CoachingDashboard()
        streamer.prefill(tokens_use[:, :8, :])
        print(f"  {'Tick':>4s} | {'Color':>5s} | {'Score':<28s} | {'Death':>5s} | {'Value':>5s} | {'Hint'}")
        print(f"  {'─'*4}─{'─'*5}─{'─'*28}─{'─'*5}─{'─'*5}─{'─'*40}")
        for t in range(8, T):
            hidden = streamer.step(tokens_use[:, t:t + 1, :])
            with torch.no_grad():
                dp = self.death_head(hidden)[:, -1, 0].mean().item()
            # Proxy value: death-risk penalty + event-based score lookahead
            val_death = -dp * 1.5
            val_event = 0.0
            token_idx = t - 1  # map to tokens_train index (tokens_use shifted by +1 for [MAP])
            if hasattr(self, "value_t_train") and self.value_t_train is not None:
                vt = self.value_t_train
                if token_idx < vt.shape[1]:
                    val_event = min(1.0, max(-1.0, vt[0, token_idx, 0].item()))
            val = val_death + val_event * 0.5
            val = max(-1.0, min(1.0, val))
            score = max(0.0, min(1.0, 1.0 - dp * 3))
            hint = XAIModule.generate_hint(score, dp, self.map_name)
            payload = dash.process_tick(tick=t, score=score, death_prob=dp, value=val, hint=hint)
            bar = "#" * int(score * 20) + "-" * (20 - int(score * 20))
            score_str = f"[{bar}] {score:.2f}"
            print(f"  {t:4d} | {payload.color:>5s} | {score_str:<28s} | {dp:5.2f} | {val:+5.2f} | {payload.hint}")
        log = self.output_dir / "dashboard_log.json"
        log.write_text(json.dumps(dash.get_history(), indent=2))
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
    p.add_argument("--player", type=str, default=None, help="Filter dashboard to specific player")
    p.add_argument("--player-num", type=int, default=None, help="Select player by number from list")
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
    # Player selection: show numbered list, filter if requested
    players = pipe.get_players()
    if players:
        print(f"\n  Players ({len(players)}):")
        for i, p in enumerate(players):
            print(f"    [{i+1}] {p}")
    if args.player_num is not None:
        pname = None
        if 1 <= args.player_num <= len(players):
            pname = players[args.player_num - 1]
        elif players:
            print(f"  [!] Invalid player number {args.player_num}")
        if pname:
            pipe.filter_to_player(pname)
            print(f"  → Filtered to player: {pname}")
    elif args.player:
        pipe.filter_to_player(args.player)
        print(f"  → Filtered to player: {args.player}")
    elif players:
        while True:
            try:
                choice = input(f"\n  Select player [1-{len(players)}] or Enter for all: ").strip()
                if not choice:
                    break
                num = int(choice)
                if 1 <= num <= len(players):
                    pname = players[num - 1]
                    pipe.filter_to_player(pname)
                    print(f"  → Filtered to player: {pname}")
                    break
                else:
                    print(f"  [!] Enter a number 1-{len(players)}")
            except ValueError:
                print(f"  [!] Enter a number 1-{len(players)}")
    pipe.build_tokenizer()
    pipe.tokenize()
    pipe.build_model()
    if args.train:
        pipe.train()
    if args.dashboard:
        pipe.run_dashboard(num_ticks=args.ticks, player_name=args.player, player_num=args.player_num)
    print(f"\n{'='*50}\n  PIPELINE COMPLETE\n  Output: {Path(args.output).resolve()}\n{'='*50}")


if __name__ == "__main__":
    main()