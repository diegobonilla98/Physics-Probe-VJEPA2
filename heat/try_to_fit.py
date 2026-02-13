import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

script_dir = Path(__file__).resolve().parent
base_dir = script_dir / "simulations"
results_dir = script_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

test_ratio = 0.2
seed = 42
ridge_lambda = 1e-2
koopman_latent_dim = 32
spatial_channels = 64
spatial_epochs = 180
spatial_lr = 3e-3

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(seed)
np.random.seed(seed)


def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def cosine_mean_np(a, b):
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    num = np.sum(a_flat * b_flat, axis=1)
    den = np.linalg.norm(a_flat, axis=1) * np.linalg.norm(b_flat, axis=1) + 1e-8
    return float(np.mean(num / den))


def fit_ridge(X_train, y_train, X_test, lam):
    mu = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train_n = (X_train - mu) / std
    X_test_n = (X_test - mu) / std
    X_train_b = np.concatenate([X_train_n, np.ones((X_train_n.shape[0], 1), dtype=np.float32)], axis=1)
    X_test_b = np.concatenate([X_test_n, np.ones((X_test_n.shape[0], 1), dtype=np.float32)], axis=1)
    ridge_identity = np.eye(X_train_b.shape[1], dtype=np.float32)
    ridge_identity[-1, -1] = 0.0
    w = np.linalg.solve(X_train_b.T @ X_train_b + lam * ridge_identity, X_train_b.T @ y_train)
    pred_train = X_train_b @ w
    pred_test = X_test_b @ w
    return pred_train, pred_test


sim_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir()])
entries = []

for sim_dir in tqdm(sim_dirs, desc="Loading simulation metadata"):
    params_path = sim_dir / "params.json"
    per_frame_path = sim_dir / "per_frame.npy"
    tokens_path = sim_dir / "tokens_spatial_fp16.npy"
    if not params_path.exists() or not per_frame_path.exists():
        continue
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    per_frame = np.load(per_frame_path).astype(np.float32)
    if per_frame.ndim != 2 or per_frame.shape[0] < 4:
        continue
    entries.append(
        {
            "name": sim_dir.name,
            "path": sim_dir,
            "alpha": float(params["alpha"]),
            "sigma": float(params["sigma"]),
            "per_frame": per_frame,
            "tokens_path": tokens_path if tokens_path.exists() else None,
        }
    )

if len(entries) < 6:
    raise ValueError("Not enough valid simulations. Run equation_sim.py and embed_sim.py first.")

perm = np.random.permutation(len(entries))
n_test = max(1, int(len(entries) * test_ratio))
test_ids = set(perm[:n_test].tolist())
train_ids = set(perm[n_test:].tolist())
train_entries = [entries[i] for i in sorted(train_ids)]
test_entries = [entries[i] for i in sorted(test_ids)]

print(f"Sims: {len(entries)} total, {len(train_entries)} train, {len(test_entries)} test")

alpha_train = np.array([e["alpha"] for e in train_entries], dtype=np.float32)
alpha_test = np.array([e["alpha"] for e in test_entries], dtype=np.float32)

X_train_sim = np.stack([e["per_frame"].mean(axis=0) for e in train_entries], axis=0).astype(np.float32)
X_test_sim = np.stack([e["per_frame"].mean(axis=0) for e in test_entries], axis=0).astype(np.float32)
pred_train_sim, pred_test_sim = fit_ridge(X_train_sim, alpha_train, X_test_sim, ridge_lambda)
alpha_probe_sim_train_r2 = r2_score_np(alpha_train, pred_train_sim)
alpha_probe_sim_test_r2 = r2_score_np(alpha_test, pred_test_sim)
alpha_probe_sim_train_mae = mae_np(alpha_train, pred_train_sim)
alpha_probe_sim_test_mae = mae_np(alpha_test, pred_test_sim)

X_train_frame = np.concatenate([e["per_frame"] for e in train_entries], axis=0).astype(np.float32)
X_test_frame = np.concatenate([e["per_frame"] for e in test_entries], axis=0).astype(np.float32)
y_train_frame = np.concatenate(
    [np.full((e["per_frame"].shape[0],), e["alpha"], dtype=np.float32) for e in train_entries], axis=0
)
y_test_frame = np.concatenate(
    [np.full((e["per_frame"].shape[0],), e["alpha"], dtype=np.float32) for e in test_entries], axis=0
)
pred_train_frame, pred_test_frame = fit_ridge(X_train_frame, y_train_frame, X_test_frame, ridge_lambda)
alpha_probe_frame_train_r2 = r2_score_np(y_train_frame, pred_train_frame)
alpha_probe_frame_test_r2 = r2_score_np(y_test_frame, pred_test_frame)
alpha_probe_frame_train_mae = mae_np(y_train_frame, pred_train_frame)
alpha_probe_frame_test_mae = mae_np(y_test_frame, pred_test_frame)

D = train_entries[0]["per_frame"].shape[1]
k = min(koopman_latent_dim, D)

train_frames = np.concatenate([e["per_frame"] for e in train_entries], axis=0).astype(np.float32)
mean_frame = train_frames.mean(axis=0, keepdims=True)
Xc = train_frames - mean_frame
_, _, vh = np.linalg.svd(Xc, full_matrices=False)
W = vh[:k].T.astype(np.float32)


def project(seq):
    return (seq - mean_frame) @ W


def reconstruct(z):
    return z @ W.T + mean_frame


Z_train_t = np.concatenate([project(e["per_frame"][:-1]) for e in train_entries], axis=0)
Z_train_next = np.concatenate([project(e["per_frame"][1:]) for e in train_entries], axis=0)
A = np.linalg.lstsq(Z_train_t, Z_train_next, rcond=None)[0].astype(np.float32)
eigvals = np.linalg.eigvals(A)
koopman_spectral_radius = float(np.max(np.abs(eigvals)))


def eval_koopman_one_step(entries_list):
    mse_list = []
    cos_list = []
    for e in entries_list:
        seq = e["per_frame"]
        z = project(seq)
        z_pred = z[:-1] @ A
        e_pred = reconstruct(z_pred)
        e_true = seq[1:]
        mse_list.append(float(np.mean((e_pred - e_true) ** 2)))
        cos_list.append(cosine_mean_np(e_pred, e_true))
    return np.array(mse_list, dtype=np.float32), np.array(cos_list, dtype=np.float32)


def eval_koopman_rollout(entries_list):
    cos_per_sim = []
    for e in entries_list:
        seq = e["per_frame"]
        z_true = project(seq)
        z_roll = np.zeros_like(z_true)
        z_roll[0] = z_true[0]
        for t in range(1, z_true.shape[0]):
            z_roll[t] = z_roll[t - 1] @ A
        e_roll = reconstruct(z_roll)
        cos_per_sim.append(cosine_mean_np(e_roll[1:], seq[1:]))
    return np.array(cos_per_sim, dtype=np.float32)


koop_train_mse, koop_train_cos1 = eval_koopman_one_step(train_entries)
koop_test_mse, koop_test_cos1 = eval_koopman_one_step(test_entries)
koop_roll_train = eval_koopman_rollout(train_entries)
koop_roll_test = eval_koopman_rollout(test_entries)

spatial_train_entries = [e for e in train_entries if e["tokens_path"] is not None]
spatial_test_entries = [e for e in test_entries if e["tokens_path"] is not None]
if len(spatial_train_entries) == 0 or len(spatial_test_entries) == 0:
    raise ValueError("Missing tokens_spatial_fp16.npy files. Run embed_sim.py first.")


def prepare_spatial_data(entries_list, channels):
    out = []
    for e in tqdm(entries_list, desc="Loading spatial tokens", leave=False):
        tok = np.load(e["tokens_path"]).astype(np.float32)
        if tok.ndim != 4 or tok.shape[0] < 4:
            continue
        c = min(channels, tok.shape[-1])
        tok = tok[..., :c]
        tok_tensor = torch.from_numpy(tok).permute(0, 3, 1, 2).contiguous()
        out.append(
            {
                "name": e["name"],
                "alpha": float(e["alpha"]),
                "tokens": tok_tensor,
            }
        )
    return out


spatial_train_data = prepare_spatial_data(spatial_train_entries, spatial_channels)
spatial_test_data = prepare_spatial_data(spatial_test_entries, spatial_channels)
if len(spatial_train_data) == 0 or len(spatial_test_data) == 0:
    raise ValueError("Spatial token tensors are empty after loading.")

effective_channels = spatial_train_data[0]["tokens"].shape[1]


class SpatialHeatHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)

    def forward(self, x, alpha):
        lap = self.depthwise(x)
        return x + alpha.view(-1, 1, 1, 1) * lap


def eval_spatial_loss(model, dataset, mode):
    model.eval()
    losses = []
    with torch.no_grad():
        for item in dataset:
            tokens = item["tokens"].to(device)
            x = tokens[:-1]
            y = tokens[1:]
            if mode == "real":
                alpha_val = float(item["alpha"])
            elif mode == "none":
                alpha_val = 1.0
            else:
                alpha_val = float(item["alpha"])
            alpha_vec = torch.full((x.shape[0],), alpha_val, device=device, dtype=torch.float32)
            pred = model(x, alpha_vec)
            losses.append(float(F.mse_loss(pred, y).item()))
    return float(np.mean(losses))


def train_spatial_model(train_data, test_data, mode, epochs, lr):
    model = SpatialHeatHead(effective_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_hist = []
    test_hist = []
    alpha_pool = [float(item["alpha"]) for item in train_data]
    for _ in tqdm(range(epochs), desc=f"Spatial train ({mode})"):
        model.train()
        order = np.random.permutation(len(train_data))
        losses = []
        for idx in order:
            item = train_data[int(idx)]
            tokens = item["tokens"].to(device)
            x = tokens[:-1]
            y = tokens[1:]
            if mode == "real":
                alpha_val = float(item["alpha"])
            elif mode == "shuffled":
                alpha_val = float(alpha_pool[np.random.randint(0, len(alpha_pool))])
            else:
                alpha_val = 1.0
            alpha_vec = torch.full((x.shape[0],), alpha_val, device=device, dtype=torch.float32)
            pred = model(x, alpha_vec)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        train_hist.append(float(np.mean(losses)))
        test_hist.append(eval_spatial_loss(model, test_data, mode))
    return model, train_hist, test_hist


def rollout_spatial_cos(model, tokens_cpu, alpha_value):
    model.eval()
    with torch.no_grad():
        tokens = tokens_cpu.to(device)
        T = tokens.shape[0]
        roll = [tokens[0:1]]
        for _ in range(1, T):
            x = roll[-1]
            alpha_vec = torch.full((1,), float(alpha_value), device=device, dtype=torch.float32)
            pred_next = model(x, alpha_vec)
            roll.append(pred_next)
        roll_t = torch.cat(roll, dim=0)
        true_t = tokens
    roll_np = roll_t[1:].detach().cpu().numpy()
    true_np = true_t[1:].detach().cpu().numpy()
    return cosine_mean_np(roll_np, true_np)


model_real, spatial_real_train_hist, spatial_real_test_hist = train_spatial_model(
    spatial_train_data, spatial_test_data, "real", spatial_epochs, spatial_lr
)
model_shuf, spatial_shuf_train_hist, spatial_shuf_test_hist = train_spatial_model(
    spatial_train_data, spatial_test_data, "shuffled", spatial_epochs, spatial_lr
)
model_none, spatial_none_train_hist, spatial_none_test_hist = train_spatial_model(
    spatial_train_data, spatial_test_data, "none", spatial_epochs, spatial_lr
)

torch.save(model_real.state_dict(), results_dir / "spatial_heat_head.pt")

spatial_roll_train = np.array(
    [rollout_spatial_cos(model_real, item["tokens"], item["alpha"]) for item in spatial_train_data],
    dtype=np.float32,
)
spatial_roll_test = np.array(
    [rollout_spatial_cos(model_real, item["tokens"], item["alpha"]) for item in spatial_test_data],
    dtype=np.float32,
)

summary_lines = [
    "heat physics extraction summary",
    "=" * 64,
    f"sims: total={len(entries)} train={len(train_entries)} test={len(test_entries)}",
    f"embedding_dim={D} koopman_latent_dim={k} spatial_channels={effective_channels}",
    "",
    "alpha linear probe (sim mean embeddings)",
    "-" * 64,
    f"train_r2={alpha_probe_sim_train_r2:.4f} test_r2={alpha_probe_sim_test_r2:.4f}",
    f"train_mae={alpha_probe_sim_train_mae:.6f} test_mae={alpha_probe_sim_test_mae:.6f}",
    "",
    "alpha linear probe (frame embeddings)",
    "-" * 64,
    f"train_r2={alpha_probe_frame_train_r2:.4f} test_r2={alpha_probe_frame_test_r2:.4f}",
    f"train_mae={alpha_probe_frame_train_mae:.6f} test_mae={alpha_probe_frame_test_mae:.6f}",
    "",
    "koopman in pca latent",
    "-" * 64,
    f"spectral_radius={koopman_spectral_radius:.6f}",
    f"one_step_mse_train={koop_train_mse.mean():.6e} one_step_mse_test={koop_test_mse.mean():.6e}",
    f"one_step_cos_train={koop_train_cos1.mean():.4f} one_step_cos_test={koop_test_cos1.mean():.4f}",
    f"rollout_cos_train={koop_roll_train.mean():.4f} +/- {koop_roll_train.std():.4f}",
    f"rollout_cos_test={koop_roll_test.mean():.4f} +/- {koop_roll_test.std():.4f}",
    "",
    "spatial token diffusion head (real alpha / shuffled alpha / no alpha)",
    "-" * 64,
    f"final_train_loss_real={spatial_real_train_hist[-1]:.6e} final_test_loss_real={spatial_real_test_hist[-1]:.6e}",
    f"final_train_loss_shuf={spatial_shuf_train_hist[-1]:.6e} final_test_loss_shuf={spatial_shuf_test_hist[-1]:.6e}",
    f"final_train_loss_none={spatial_none_train_hist[-1]:.6e} final_test_loss_none={spatial_none_test_hist[-1]:.6e}",
    f"rollout_cos_train={spatial_roll_train.mean():.4f} +/- {spatial_roll_train.std():.4f}",
    f"rollout_cos_test={spatial_roll_test.mean():.4f} +/- {spatial_roll_test.std():.4f}",
]

with open(results_dir / "fit_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

metrics_json = {
    "alpha_probe_sim": {
        "train_r2": float(alpha_probe_sim_train_r2),
        "test_r2": float(alpha_probe_sim_test_r2),
        "train_mae": float(alpha_probe_sim_train_mae),
        "test_mae": float(alpha_probe_sim_test_mae),
    },
    "alpha_probe_frame": {
        "train_r2": float(alpha_probe_frame_train_r2),
        "test_r2": float(alpha_probe_frame_test_r2),
        "train_mae": float(alpha_probe_frame_train_mae),
        "test_mae": float(alpha_probe_frame_test_mae),
    },
    "koopman": {
        "latent_dim": int(k),
        "spectral_radius": float(koopman_spectral_radius),
        "one_step_mse_train": float(koop_train_mse.mean()),
        "one_step_mse_test": float(koop_test_mse.mean()),
        "one_step_cos_train": float(koop_train_cos1.mean()),
        "one_step_cos_test": float(koop_test_cos1.mean()),
        "rollout_cos_train_mean": float(koop_roll_train.mean()),
        "rollout_cos_train_std": float(koop_roll_train.std()),
        "rollout_cos_test_mean": float(koop_roll_test.mean()),
        "rollout_cos_test_std": float(koop_roll_test.std()),
    },
    "spatial": {
        "channels": int(effective_channels),
        "real_final_train_loss": float(spatial_real_train_hist[-1]),
        "real_final_test_loss": float(spatial_real_test_hist[-1]),
        "shuffled_final_train_loss": float(spatial_shuf_train_hist[-1]),
        "shuffled_final_test_loss": float(spatial_shuf_test_hist[-1]),
        "none_final_train_loss": float(spatial_none_train_hist[-1]),
        "none_final_test_loss": float(spatial_none_test_hist[-1]),
        "rollout_cos_train_mean": float(spatial_roll_train.mean()),
        "rollout_cos_train_std": float(spatial_roll_train.std()),
        "rollout_cos_test_mean": float(spatial_roll_test.mean()),
        "rollout_cos_test_std": float(spatial_roll_test.std()),
    },
}
with open(results_dir / "fit_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_json, f, indent=2)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].scatter(alpha_test, pred_test_sim, s=40, color="tab:blue")
min_alpha = float(min(alpha_test.min(), pred_test_sim.min()))
max_alpha = float(max(alpha_test.max(), pred_test_sim.max()))
axes[0, 0].plot([min_alpha, max_alpha], [min_alpha, max_alpha], color="black", linewidth=1)
axes[0, 0].set_xlabel("true alpha")
axes[0, 0].set_ylabel("predicted alpha")
axes[0, 0].set_title("alpha probe on held-out sims")

axes[0, 1].bar(np.arange(len(koop_roll_train)), koop_roll_train, color="steelblue", label="train")
axes[0, 1].bar(
    np.arange(len(koop_roll_train), len(koop_roll_train) + len(koop_roll_test)),
    koop_roll_test,
    color="coral",
    label="test",
)
axes[0, 1].set_xlabel("simulation")
axes[0, 1].set_ylabel("rollout cosine")
axes[0, 1].set_title("koopman rollout per simulation")
axes[0, 1].legend()

axes[1, 0].plot(spatial_real_train_hist, label="real train")
axes[1, 0].plot(spatial_real_test_hist, label="real test")
axes[1, 0].plot(spatial_shuf_train_hist, label="shuffled train", alpha=0.7)
axes[1, 0].plot(spatial_shuf_test_hist, label="shuffled test", alpha=0.7)
axes[1, 0].plot(spatial_none_train_hist, label="no-alpha train", alpha=0.7)
axes[1, 0].plot(spatial_none_test_hist, label="no-alpha test", alpha=0.7)
axes[1, 0].set_yscale("log")
axes[1, 0].set_xlabel("epoch")
axes[1, 0].set_ylabel("mse")
axes[1, 0].set_title("spatial token training curves")
axes[1, 0].legend(fontsize=8)

axes[1, 1].bar(np.arange(len(spatial_roll_train)), spatial_roll_train, color="steelblue", label="train")
axes[1, 1].bar(
    np.arange(len(spatial_roll_train), len(spatial_roll_train) + len(spatial_roll_test)),
    spatial_roll_test,
    color="coral",
    label="test",
)
axes[1, 1].set_xlabel("simulation")
axes[1, 1].set_ylabel("rollout cosine")
axes[1, 1].set_title("spatial token rollout per simulation")
axes[1, 1].legend()

plt.tight_layout()
plt.savefig(results_dir / "fit_results.png", dpi=150)
plt.show()

print(f"Saved results to {results_dir}")
