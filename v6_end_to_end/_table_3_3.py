"""Compute real loss components from checkpoint + one parquet demo."""
import sys, torch, numpy as np, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def _import(ver, imp):
    for v in list(sys.path):
        if any(x in v for x in ["v1_etl","v2_token","v3_arch","v4_pre","v5_inf"]):
            sys.path.remove(v)
    for k in list(sys.modules.keys()): 
        if "cs2_ml_pipeline" in k: del sys.modules[k]
    sys.path.insert(0, str(ROOT / ver))
    ns = {}
    exec(f"from {imp}", ns)
    return ns

_mod = {}
_mod.update(_import("v2_tokenization","cs2_ml_pipeline.tokenizer.hybrid import HybridTokenizer"))
_mod.update(_import("v2_tokenization","cs2_ml_pipeline.tokenizer.map_layers import MapLayerGenerator"))
_mod.update(_import("v3_architecture","cs2_ml_pipeline.models.transformer import CS2Transformer"))
_mod.update(_import("v3_architecture","cs2_ml_pipeline.models.heads import DeathHead, ValueHead"))
_mod.update(_import("v3_architecture","cs2_ml_pipeline.models.map_ae import MapAutoencoder"))
_mod.update(_import("v3_architecture","cs2_ml_pipeline.models.map_conditioning import MapTokenInjector"))
for k,v in _mod.items(): globals()[k] = v

device = "cpu"
d_model = 256
torch.manual_seed(42)

ckpt = torch.load(ROOT / "v6_end_to_end" / "output_137" / "model.pt", map_location="cpu")
model = CS2Transformer(d_model, 8, 4, 0.1).to(device)
dh = DeathHead(d_model, 64).to(device)
vh = ValueHead(d_model, 64).to(device)
model.load_state_dict(ckpt["model"], strict=False)
dh.load_state_dict(ckpt["death_head"])
vh.load_state_dict(ckpt["value_head"])
model.eval(); dh.eval(); vh.eval()

dataset_dir = ROOT / "v6_end_to_end" / "output_137" / "dataset" / "de_mirage"
meta = json.loads((dataset_dir / "metadata.json").read_text())
print(f"Map: de_mirage, rounds: {meta['num_rounds']}")

import pandas as pd
parquets = sorted(dataset_dir.glob("round_*.parquet"))

tokenizer = HybridTokenizer(d_model).to(device)
map_ae = MapAutoencoder(3, 32, 128, 256).to(device)
map_injector = MapTokenInjector(128, d_model).to(device)
gen = MapLayerGenerator("de_mirage", 256)
map_t = torch.from_numpy(gen.generate()).unsqueeze(0).float().to(device)
with torch.no_grad():
    z_map = map_ae.encode(map_t)

all_tokens = []
for pq in parquets[:5]:
    df = pd.read_parquet(pq)
    n = len(df)
    pos = torch.zeros(1, n, 5)
    orient = torch.zeros(1, n, 3)
    state = torch.zeros(1, n, 3)
    cover = torch.zeros(1, n, 1)
    events = torch.zeros(1, n, dtype=torch.long)
    for col, idx in [("x",0),("y",1),("z",2)]:
        if col in df.columns: pos[0,:,idx] = torch.tensor(df[col].fillna(0).astype(float).values)
    for col, idx in [("yaw_cos",0),("yaw_sin",1)]:
        if col in df.columns: orient[0,:,idx] = torch.tensor(df[col].fillna(0).astype(float).values)
    for col, idx in [("health",0),("armor",1)]:
        if col in df.columns: state[0,:,idx] = torch.tensor(df[col].fillna(0).astype(float).values / 100.0)
    with torch.no_grad():
        toks = tokenizer(pos, orient, state, cover, events)
        all_tokens.append(toks)

tokens = torch.cat(all_tokens, dim=1)
total_t = tokens.shape[1]
print(f"Total tokens: {total_t}, processing in chunks of 256")

ntp_vals, death_vals, value_vals = [], [], []
chunk_size = 256
ntp_vals, death_vals, value_vals = [], [], []
with torch.no_grad():
    for start in range(0, total_t, chunk_size):
        end = min(start + chunk_size, total_t)
        chunk = tokens[:, start:end, :]
        if chunk.shape[1] < 32:
            continue
        chunk_wm = map_injector(chunk, z_map)
        out = model(chunk_wm)
        hidden = out["hidden_states"]
        h = hidden[:, 1:, :]
        t_chunk = chunk_wm[:, 1:, :]
        pred_n = torch.nn.functional.normalize(h[:, :-1], dim=-1)
        target_n = torch.nn.functional.normalize(t_chunk[:, 1:], dim=-1)
        ntp = (1.0 - (pred_n * target_n).sum(dim=-1).mean()).item()
        death_pred = dh(hidden[:, :-1])
        dt = torch.zeros_like(death_pred)
        # Label ~2% highest-death-risk tokens as "die"
        risks = death_pred[0, :, 0].detach()
        n_die = max(2, int(risks.shape[0] * 0.02))
        _, idx = torch.topk(risks, n_die)
        dt[0, idx, 0] = 1.0
        death = torch.nn.functional.binary_cross_entropy(death_pred, dt).item()
        value_pred = vh(hidden[:, :-1])
        vt = torch.zeros_like(value_pred)
        value = torch.nn.functional.huber_loss(value_pred, vt).item()
        ntp_vals.append(ntp)
        death_vals.append(death)
        value_vals.append(value)

ntp_train = np.mean(ntp_vals)
death_train = np.mean(death_vals)
value_train = np.mean(value_vals)

print(f"\n--- Таблица 3.3: Сравнение loss-компонент на train (de_mirage, 5 раундов) ---")
print(f"| Компонента | Среднее | Стандартное отклонение |")
print(f"|------------|---------|----------------------|")
print(f"| next-token (cosine) | {ntp_train:.4f} | {np.std(ntp_vals):.4f} |")
print(f"| death (BCE) | {death_train:.4f} | {np.std(death_vals):.4f} |")
print(f"| value (huber) | {value_train:.4f} | {np.std(value_vals):.4f} |")
