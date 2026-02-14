import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")

script_dir = Path(__file__).resolve().parent
base_dir = script_dir / "simulations"
results_dir = script_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

test_ratio = 0.2
seed = 42
ridge_lambda = 1e-2
koopman_latent_dim = 32

rng = np.random.RandomState(seed)

eq_to_id = {"diffusion": 0, "wave": 1}


def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def total_variance_r2(y_true, y_pred):
    y_mu = np.mean(y_true, axis=0, keepdims=True)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mu) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


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
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    reg_eye = np.eye(X_train_b.shape[1], dtype=np.float32)
    reg_eye[-1, -1] = 0.0
    w = np.linalg.solve(X_train_b.T @ X_train_b + lam * reg_eye, X_train_b.T @ y_train)
    pred_train = X_train_b @ w
    pred_test = X_test_b @ w
    if pred_train.shape[1] == 1:
        pred_train = pred_train[:, 0]
        pred_test = pred_test[:, 0]
    return pred_train.astype(np.float32), pred_test.astype(np.float32)


def fit_linear_classifier(X_train, y_train_binary, X_test, lam):
    y_signed = (2 * y_train_binary.astype(np.float32) - 1.0).astype(np.float32)
    score_train, score_test = fit_ridge(X_train, y_signed, X_test, lam)
    return score_train, score_test


def classification_metrics(y_true, score):
    y_pred = (score >= 0.0).astype(np.int64)
    acc = float(np.mean(y_pred == y_true))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    precision = float(tp / (tp + fp + 1e-12))
    recall = float(tp / (tp + fn + 1e-12))
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def stratified_split(entries, test_frac, local_rng):
    by_eq = {}
    for idx, e in enumerate(entries):
        by_eq.setdefault(e["equation"], []).append(idx)
    test_ids = []
    train_ids = []
    for _, idxs in by_eq.items():
        idxs = np.array(idxs, dtype=np.int64)
        local_rng.shuffle(idxs)
        n_test = max(1, int(round(len(idxs) * test_frac)))
        test_ids.extend(idxs[:n_test].tolist())
        train_ids.extend(idxs[n_test:].tolist())
    return sorted(train_ids), sorted(test_ids)


def fit_koopman_first_order(train_entries, test_entries, latent_dim):
    train_frames = np.concatenate([e["per_frame"] for e in train_entries], axis=0).astype(np.float32)
    mean_frame = train_frames.mean(axis=0, keepdims=True)
    Xc = train_frames - mean_frame
    _, _, vh = np.linalg.svd(Xc, full_matrices=False)
    k = min(latent_dim, vh.shape[0], vh.shape[1])
    W = vh[:k].T.astype(np.float32)

    def project(seq):
        return (seq - mean_frame) @ W

    def reconstruct(z):
        return z @ W.T + mean_frame

    Z_t = np.concatenate([project(e["per_frame"][:-1]) for e in train_entries], axis=0)
    Z_n = np.concatenate([project(e["per_frame"][1:]) for e in train_entries], axis=0)
    A = np.linalg.lstsq(Z_t, Z_n, rcond=None)[0].astype(np.float32)

    def eval_rollout(entries):
        cos_vals = []
        mse_vals = []
        for e in entries:
            z_true = project(e["per_frame"])
            z_roll = np.zeros_like(z_true)
            z_roll[0] = z_true[0]
            for t in range(1, z_true.shape[0]):
                z_roll[t] = z_roll[t - 1] @ A
            rec = reconstruct(z_roll)
            cos_vals.append(cosine_mean_np(rec[1:], e["per_frame"][1:]))
            mse_vals.append(float(np.mean((rec[1:] - e["per_frame"][1:]) ** 2)))
        return {
            "rollout_cos_mean": float(np.mean(cos_vals)),
            "rollout_cos_std": float(np.std(cos_vals)),
            "rollout_mse_mean": float(np.mean(mse_vals)),
        }

    return {
        "latent_dim": int(k),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(A)))),
        "train": eval_rollout(train_entries),
        "test": eval_rollout(test_entries),
    }


def fit_koopman_second_order(train_entries, test_entries, latent_dim):
    train_frames = np.concatenate([e["per_frame"] for e in train_entries], axis=0).astype(np.float32)
    mean_frame = train_frames.mean(axis=0, keepdims=True)
    Xc = train_frames - mean_frame
    _, _, vh = np.linalg.svd(Xc, full_matrices=False)
    k = min(latent_dim, vh.shape[0], vh.shape[1])
    W = vh[:k].T.astype(np.float32)

    def project(seq):
        return (seq - mean_frame) @ W

    def reconstruct(z):
        return z @ W.T + mean_frame

    X_list = []
    Y_list = []
    for e in train_entries:
        z = project(e["per_frame"])
        if z.shape[0] < 3:
            continue
        z_tm1 = z[:-2]
        z_t = z[1:-1]
        z_tp1 = z[2:]
        X_list.append(np.concatenate([z_t, z_tm1], axis=1))
        Y_list.append(z_tp1)
    X_train = np.concatenate(X_list, axis=0)
    Y_train = np.concatenate(Y_list, axis=0)
    B = np.linalg.lstsq(X_train, Y_train, rcond=None)[0].astype(np.float32)

    def eval_rollout(entries):
        cos_vals = []
        mse_vals = []
        for e in entries:
            z_true = project(e["per_frame"])
            if z_true.shape[0] < 3:
                continue
            z_roll = np.zeros_like(z_true)
            z_roll[0] = z_true[0]
            z_roll[1] = z_true[1]
            for t in range(1, z_true.shape[0] - 1):
                x_in = np.concatenate([z_roll[t], z_roll[t - 1]], axis=0)
                z_roll[t + 1] = x_in @ B
            rec = reconstruct(z_roll)
            cos_vals.append(cosine_mean_np(rec[2:], e["per_frame"][2:]))
            mse_vals.append(float(np.mean((rec[2:] - e["per_frame"][2:]) ** 2)))
        return {
            "rollout_cos_mean": float(np.mean(cos_vals)),
            "rollout_cos_std": float(np.std(cos_vals)),
            "rollout_mse_mean": float(np.mean(mse_vals)),
        }

    return {
        "latent_dim": int(k),
        "train": eval_rollout(train_entries),
        "test": eval_rollout(test_entries),
    }


sim_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("sim_")])
entries = []
for sim_dir in tqdm(sim_dirs, desc="Loading wave/diffusion sims"):
    params_path = sim_dir / "params.json"
    emb_path = sim_dir / "per_frame.npy"
    if not params_path.exists() or not emb_path.exists():
        continue
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    eq = str(params.get("equation", ""))
    if eq not in eq_to_id:
        continue
    per_frame = np.load(emb_path).astype(np.float32)
    if per_frame.ndim != 2 or per_frame.shape[0] < 4:
        continue
    entries.append(
        {
            "name": sim_dir.name,
            "equation": eq,
            "per_frame": per_frame,
            "c": float(params.get("c", 0.0)),
            "gamma": float(params.get("gamma", 0.0)),
            "alpha": float(params.get("alpha", 0.0)),
        }
    )

if len(entries) < 10:
    raise ValueError("Not enough embedded simulations. Run equation_sim.py then embed_sim.py first.")

train_ids, test_ids = stratified_split(entries, test_ratio, rng)
train_entries = [entries[i] for i in train_ids]
test_entries = [entries[i] for i in test_ids]

print(f"Sims: total={len(entries)} train={len(train_entries)} test={len(test_entries)}")

X_train_sim = np.stack([e["per_frame"].mean(axis=0) for e in train_entries], axis=0).astype(np.float32)
X_test_sim = np.stack([e["per_frame"].mean(axis=0) for e in test_entries], axis=0).astype(np.float32)
y_train_eq = np.array([eq_to_id[e["equation"]] for e in train_entries], dtype=np.int64)
y_test_eq = np.array([eq_to_id[e["equation"]] for e in test_entries], dtype=np.int64)
y_train_params = np.array([[e["c"], e["gamma"], e["alpha"]] for e in train_entries], dtype=np.float32)
y_test_params = np.array([[e["c"], e["gamma"], e["alpha"]] for e in test_entries], dtype=np.float32)

# Equation classification.
score_train_sim, score_test_sim = fit_linear_classifier(X_train_sim, y_train_eq, X_test_sim, ridge_lambda)
eq_cls_sim_train = classification_metrics(y_train_eq, score_train_sim)
eq_cls_sim_test = classification_metrics(y_test_eq, score_test_sim)

X_train_frame = np.concatenate([e["per_frame"] for e in train_entries], axis=0).astype(np.float32)
X_test_frame = np.concatenate([e["per_frame"] for e in test_entries], axis=0).astype(np.float32)
y_train_frame_eq = np.concatenate(
    [np.full((e["per_frame"].shape[0],), eq_to_id[e["equation"]], dtype=np.int64) for e in train_entries], axis=0
)
y_test_frame_eq = np.concatenate(
    [np.full((e["per_frame"].shape[0],), eq_to_id[e["equation"]], dtype=np.int64) for e in test_entries], axis=0
)
score_train_frame, score_test_frame = fit_linear_classifier(X_train_frame, y_train_frame_eq, X_test_frame, ridge_lambda)
eq_cls_frame_train = classification_metrics(y_train_frame_eq, score_train_frame)
eq_cls_frame_test = classification_metrics(y_test_frame_eq, score_test_frame)

# Unified parameter regression [c, gamma, alpha].
pred_train_all, pred_test_all = fit_ridge(X_train_sim, y_train_params, X_test_sim, ridge_lambda)
param_all_train_r2 = total_variance_r2(y_train_params, pred_train_all)
param_all_test_r2 = total_variance_r2(y_test_params, pred_test_all)

# Equation-specific parameter probes.
train_wave = [e for e in train_entries if e["equation"] == "wave"]
test_wave = [e for e in test_entries if e["equation"] == "wave"]
train_diff = [e for e in train_entries if e["equation"] == "diffusion"]
test_diff = [e for e in test_entries if e["equation"] == "diffusion"]

X_wave_train = np.stack([e["per_frame"].mean(axis=0) for e in train_wave], axis=0).astype(np.float32)
X_wave_test = np.stack([e["per_frame"].mean(axis=0) for e in test_wave], axis=0).astype(np.float32)
y_wave_train = np.array([[e["c"], e["gamma"]] for e in train_wave], dtype=np.float32)
y_wave_test = np.array([[e["c"], e["gamma"]] for e in test_wave], dtype=np.float32)
pred_wave_train, pred_wave_test = fit_ridge(X_wave_train, y_wave_train, X_wave_test, ridge_lambda)

wave_c_r2_test = r2_score_np(y_wave_test[:, 0], pred_wave_test[:, 0])
wave_gamma_r2_test = r2_score_np(y_wave_test[:, 1], pred_wave_test[:, 1])
wave_c_mae_test = mae_np(y_wave_test[:, 0], pred_wave_test[:, 0])
wave_gamma_mae_test = mae_np(y_wave_test[:, 1], pred_wave_test[:, 1])

X_diff_train = np.stack([e["per_frame"].mean(axis=0) for e in train_diff], axis=0).astype(np.float32)
X_diff_test = np.stack([e["per_frame"].mean(axis=0) for e in test_diff], axis=0).astype(np.float32)
y_diff_train = np.array([e["alpha"] for e in train_diff], dtype=np.float32)
y_diff_test = np.array([e["alpha"] for e in test_diff], dtype=np.float32)
pred_diff_train, pred_diff_test = fit_ridge(X_diff_train, y_diff_train, X_diff_test, ridge_lambda)
diff_alpha_r2_test = r2_score_np(y_diff_test, pred_diff_test)
diff_alpha_mae_test = mae_np(y_diff_test, pred_diff_test)

# Koopman diagnostics.
koopman_shared_first = fit_koopman_first_order(train_entries, test_entries, koopman_latent_dim)
koopman_wave_first = fit_koopman_first_order(train_wave, test_wave, koopman_latent_dim)
koopman_wave_second = fit_koopman_second_order(train_wave, test_wave, koopman_latent_dim)
koopman_diff_first = fit_koopman_first_order(train_diff, test_diff, koopman_latent_dim)

metrics = {
    "setup": {
        "n_total": int(len(entries)),
        "n_train": int(len(train_entries)),
        "n_test": int(len(test_entries)),
        "seed": int(seed),
        "test_ratio": float(test_ratio),
        "ridge_lambda": float(ridge_lambda),
        "koopman_latent_dim": int(koopman_latent_dim),
    },
    "equation_classification": {
        "sim_mean_train": eq_cls_sim_train,
        "sim_mean_test": eq_cls_sim_test,
        "frame_train": eq_cls_frame_train,
        "frame_test": eq_cls_frame_test,
    },
    "parameter_regression": {
        "all_params_train_r2_total": float(param_all_train_r2),
        "all_params_test_r2_total": float(param_all_test_r2),
        "wave_c_r2_test": float(wave_c_r2_test),
        "wave_gamma_r2_test": float(wave_gamma_r2_test),
        "wave_c_mae_test": float(wave_c_mae_test),
        "wave_gamma_mae_test": float(wave_gamma_mae_test),
        "diffusion_alpha_r2_test": float(diff_alpha_r2_test),
        "diffusion_alpha_mae_test": float(diff_alpha_mae_test),
    },
    "koopman_shared_first_order": koopman_shared_first,
    "koopman_wave_first_order": koopman_wave_first,
    "koopman_wave_second_order": koopman_wave_second,
    "koopman_diffusion_first_order": koopman_diff_first,
}

with open(results_dir / "fit_metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

summary_lines = [
    "wave vs diffusion probe summary",
    "=" * 72,
    f"sims: total={len(entries)} train={len(train_entries)} test={len(test_entries)}",
    "",
    "equation classification",
    "-" * 72,
    f"sim-mean acc train/test: {eq_cls_sim_train['accuracy']:.4f} / {eq_cls_sim_test['accuracy']:.4f}",
    f"frame acc train/test: {eq_cls_frame_train['accuracy']:.4f} / {eq_cls_frame_test['accuracy']:.4f}",
    "",
    "parameter probes",
    "-" * 72,
    f"all-param total R2 train/test: {param_all_train_r2:.4f} / {param_all_test_r2:.4f}",
    f"wave c R2={wave_c_r2_test:.4f} gamma R2={wave_gamma_r2_test:.4f}",
    f"wave c MAE={wave_c_mae_test:.4f} gamma MAE={wave_gamma_mae_test:.4f}",
    f"diffusion alpha R2={diff_alpha_r2_test:.4f} MAE={diff_alpha_mae_test:.4f}",
    "",
    "koopman",
    "-" * 72,
    (
        "shared first-order test cos="
        f"{koopman_shared_first['test']['rollout_cos_mean']:.4f} +/- "
        f"{koopman_shared_first['test']['rollout_cos_std']:.4f}"
    ),
    (
        "wave first-order test cos="
        f"{koopman_wave_first['test']['rollout_cos_mean']:.4f} +/- "
        f"{koopman_wave_first['test']['rollout_cos_std']:.4f}"
    ),
    (
        "wave second-order test cos="
        f"{koopman_wave_second['test']['rollout_cos_mean']:.4f} +/- "
        f"{koopman_wave_second['test']['rollout_cos_std']:.4f}"
    ),
    (
        "diffusion first-order test cos="
        f"{koopman_diff_first['test']['rollout_cos_mean']:.4f} +/- "
        f"{koopman_diff_first['test']['rollout_cos_std']:.4f}"
    ),
]

with open(results_dir / "fit_summary.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

# Quick plot panel for diagnostics.
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0, 0].hist(score_test_sim[y_test_eq == 0], bins=12, alpha=0.7, label="diffusion")
axes[0, 0].hist(score_test_sim[y_test_eq == 1], bins=12, alpha=0.7, label="wave")
axes[0, 0].axvline(0.0, color="black", linewidth=1)
axes[0, 0].set_title("Equation score (sim-mean test)")
axes[0, 0].set_xlabel("linear score")
axes[0, 0].legend()

axes[0, 1].scatter(y_wave_test[:, 0], pred_wave_test[:, 0], s=24, label="c")
axes[0, 1].scatter(y_wave_test[:, 1], pred_wave_test[:, 1], s=24, label="gamma")
axes[0, 1].set_title("Wave parameter probe (test)")
axes[0, 1].set_xlabel("true")
axes[0, 1].set_ylabel("pred")
axes[0, 1].legend()

axes[1, 0].scatter(y_diff_test, pred_diff_test, s=28, color="tab:orange")
mn = float(min(y_diff_test.min(), pred_diff_test.min()))
mx = float(max(y_diff_test.max(), pred_diff_test.max()))
axes[1, 0].plot([mn, mx], [mn, mx], color="black", linewidth=1)
axes[1, 0].set_title("Diffusion alpha probe (test)")
axes[1, 0].set_xlabel("true alpha")
axes[1, 0].set_ylabel("pred alpha")

bar_names = ["shared-1st", "wave-1st", "wave-2nd", "diff-1st"]
bar_vals = [
    koopman_shared_first["test"]["rollout_cos_mean"],
    koopman_wave_first["test"]["rollout_cos_mean"],
    koopman_wave_second["test"]["rollout_cos_mean"],
    koopman_diff_first["test"]["rollout_cos_mean"],
]
axes[1, 1].bar(np.arange(len(bar_names)), bar_vals, color="steelblue")
axes[1, 1].set_xticks(np.arange(len(bar_names)))
axes[1, 1].set_xticklabels(bar_names, rotation=20, ha="right")
axes[1, 1].set_ylim(0.0, 1.0)
axes[1, 1].set_title("Koopman rollout cosine (test)")

plt.tight_layout()
plt.savefig(results_dir / "fit_results.png", dpi=150)
plt.close(fig)

print("\n".join(summary_lines))
print(f"\nSaved results to {results_dir}")
