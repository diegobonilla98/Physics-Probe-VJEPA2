import argparse
import json
import math
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def parse_list_arg(raw, cast_fn):
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    return [cast_fn(p) for p in parts]


def r2_score_np(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def mae_np(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def cosine_mean_np(a, b):
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    num = np.sum(a_flat * b_flat, axis=1)
    den = np.linalg.norm(a_flat, axis=1) * np.linalg.norm(b_flat, axis=1) + 1e-8
    return float(np.mean(num / den))


def total_variance_r2(y_true, y_pred):
    mu = y_true.mean(axis=0, keepdims=True)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - mu) ** 2))
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def fit_ridge(X_train, y_train, X_test, lam):
    mu = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    X_train_n = (X_train - mu) / std
    X_test_n = (X_test - mu) / std
    X_train_b = np.concatenate([X_train_n, np.ones((X_train_n.shape[0], 1), dtype=np.float32)], axis=1)
    X_test_b = np.concatenate([X_test_n, np.ones((X_test_n.shape[0], 1), dtype=np.float32)], axis=1)
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    ridge_identity = np.eye(X_train_b.shape[1], dtype=np.float32)
    ridge_identity[-1, -1] = 0.0
    w = np.linalg.solve(X_train_b.T @ X_train_b + lam * ridge_identity, X_train_b.T @ y_train)
    pred_train = X_train_b @ w
    pred_test = X_test_b @ w
    if pred_train.shape[1] == 1:
        pred_train = pred_train[:, 0]
        pred_test = pred_test[:, 0]
    return pred_train.astype(np.float32), pred_test.astype(np.float32)


def stratified_split_indices(alpha, test_ratio, rng, n_bins=5):
    alpha = np.asarray(alpha, dtype=np.float32)
    n = alpha.shape[0]
    n_test_target = max(1, int(round(n * float(test_ratio))))
    if n <= 2:
        ids = np.arange(n)
        rng.shuffle(ids)
        return ids[1:].tolist(), ids[:1].tolist()

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(alpha, quantiles)
    edges = np.unique(edges)
    if edges.shape[0] <= 2:
        ids = np.arange(n)
        rng.shuffle(ids)
        return ids[n_test_target:].tolist(), ids[:n_test_target].tolist()

    bin_ids = np.digitize(alpha, edges[1:-1], right=False)
    test_idx = []
    for b in np.unique(bin_ids):
        ids = np.where(bin_ids == b)[0]
        ids = ids.copy()
        rng.shuffle(ids)
        take = max(1, int(round(ids.shape[0] * test_ratio)))
        test_idx.extend(ids[:take].tolist())
    test_idx = sorted(set(test_idx))

    if len(test_idx) < n_test_target:
        remaining = [i for i in range(n) if i not in test_idx]
        rng.shuffle(remaining)
        need = n_test_target - len(test_idx)
        test_idx.extend(remaining[:need])
    elif len(test_idx) > n_test_target:
        rng.shuffle(test_idx)
        test_idx = test_idx[:n_test_target]

    test_set = set(test_idx)
    train_idx = [i for i in range(n) if i not in test_set]
    return train_idx, sorted(test_idx)


def build_group_folds(n_items, n_folds, rng):
    ids = np.arange(n_items)
    ids = ids.copy()
    rng.shuffle(ids)
    folds = np.array_split(ids, n_folds)
    out = []
    for i in range(n_folds):
        val_ids = folds[i].tolist()
        train_ids = np.concatenate([folds[j] for j in range(n_folds) if j != i]).tolist()
        out.append((train_ids, val_ids))
    return out


def load_frames_float(sim_dir):
    frames_dir = sim_dir / "frames"
    frame_paths = sorted(
        p
        for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )
    if not frame_paths:
        raise ValueError(f"No frames found in {frames_dir}")

    all_frames = np.stack([iio.imread(p) for p in frame_paths], axis=0)
    if all_frames.ndim == 3:
        all_frames = np.repeat(all_frames[..., None], 3, axis=-1)
    if all_frames.shape[-1] == 4:
        all_frames = all_frames[..., :3]
    x = torch.from_numpy(all_frames).float() / 255.0
    return x.permute(0, 3, 1, 2).contiguous()


def center_square_crop(x):
    _, _, h, w = x.shape
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return x[:, :, y0 : y0 + side, x0 : x0 + side]


def extract_pixel_features(x, mode):
    mode = mode.strip().lower()
    if mode.startswith("rgb"):
        grid = int(mode.replace("rgb", ""))
        z = F.interpolate(x, size=(grid, grid), mode="area")
        return z.permute(0, 2, 3, 1).contiguous().view(z.shape[0], -1).cpu().numpy().astype(np.float32)

    if mode.startswith("gray"):
        grid = int(mode.replace("gray", ""))
        gray = x.mean(dim=1, keepdim=True)
        z = F.interpolate(gray, size=(grid, grid), mode="area")
        return z.view(z.shape[0], -1).cpu().numpy().astype(np.float32)

    if mode == "crop_patch16":
        z = center_square_crop(x)
        z = F.interpolate(z, size=(256, 256), mode="bilinear", align_corners=False)
        z = F.avg_pool2d(z, kernel_size=16, stride=16)
        return z.permute(0, 2, 3, 1).contiguous().view(z.shape[0], -1).cpu().numpy().astype(np.float32)

    raise ValueError(f"Unknown pixel mode: {mode}")


def make_sim_matrix(entries, key):
    return np.stack([e[key].mean(axis=0) for e in entries], axis=0).astype(np.float32)


def make_frame_matrix(entries, key):
    return np.concatenate([e[key] for e in entries], axis=0).astype(np.float32)


def make_frame_alpha(entries):
    return np.concatenate(
        [np.full((e["embed"].shape[0],), e["alpha"], dtype=np.float32) for e in entries],
        axis=0,
    )


def tune_ridge_sim(train_entries, key, lam_grid, folds, rng):
    X_all = make_sim_matrix(train_entries, key)
    y_all = np.array([e["alpha"] for e in train_entries], dtype=np.float32)
    if len(train_entries) < 4:
        return float(lam_grid[0])

    fold_defs = build_group_folds(len(train_entries), max(2, min(folds, len(train_entries))), rng)
    best_lam = float(lam_grid[0])
    best_score = -1e18
    for lam in lam_grid:
        scores = []
        for tr_ids, va_ids in fold_defs:
            pred_tr, pred_va = fit_ridge(X_all[tr_ids], y_all[tr_ids], X_all[va_ids], lam)
            _ = pred_tr
            scores.append(r2_score_np(y_all[va_ids], pred_va))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_lam = float(lam)
    return best_lam


def tune_ridge_frame(train_entries, key, lam_grid, folds, rng):
    if len(train_entries) < 4:
        return float(lam_grid[0])
    fold_defs = build_group_folds(len(train_entries), max(2, min(folds, len(train_entries))), rng)
    best_lam = float(lam_grid[0])
    best_score = -1e18
    for lam in lam_grid:
        scores = []
        for tr_ids, va_ids in fold_defs:
            tr_entries = [train_entries[i] for i in tr_ids]
            va_entries = [train_entries[i] for i in va_ids]
            X_tr = make_frame_matrix(tr_entries, key)
            X_va = make_frame_matrix(va_entries, key)
            y_tr = make_frame_alpha(tr_entries)
            y_va = make_frame_alpha(va_entries)
            pred_tr, pred_va = fit_ridge(X_tr, y_tr, X_va, lam)
            _ = pred_tr
            scores.append(r2_score_np(y_va, pred_va))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_lam = float(lam)
    return best_lam


def tune_ridge_multitarget(train_entries, in_key, out_key, lam_grid, folds, rng):
    if len(train_entries) < 4:
        return float(lam_grid[0])
    fold_defs = build_group_folds(len(train_entries), max(2, min(folds, len(train_entries))), rng)
    best_lam = float(lam_grid[0])
    best_score = -1e18
    for lam in lam_grid:
        scores = []
        for tr_ids, va_ids in fold_defs:
            tr_entries = [train_entries[i] for i in tr_ids]
            va_entries = [train_entries[i] for i in va_ids]
            X_tr = make_frame_matrix(tr_entries, in_key)
            X_va = make_frame_matrix(va_entries, in_key)
            Y_tr = make_frame_matrix(tr_entries, out_key)
            Y_va = make_frame_matrix(va_entries, out_key)
            pred_tr, pred_va = fit_ridge(X_tr, Y_tr, X_va, lam)
            _ = pred_tr
            scores.append(total_variance_r2(Y_va, pred_va))
        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_lam = float(lam)
    return best_lam


def run_alpha_probes(train_entries, test_entries, key, lam_grid, folds, seed):
    rng = np.random.RandomState(seed)
    lam_sim = tune_ridge_sim(train_entries, key, lam_grid, folds, rng)
    lam_frame = tune_ridge_frame(train_entries, key, lam_grid, folds, rng)

    alpha_train = np.array([e["alpha"] for e in train_entries], dtype=np.float32)
    alpha_test = np.array([e["alpha"] for e in test_entries], dtype=np.float32)

    X_train_sim = make_sim_matrix(train_entries, key)
    X_test_sim = make_sim_matrix(test_entries, key)
    pred_train_sim, pred_test_sim = fit_ridge(X_train_sim, alpha_train, X_test_sim, lam_sim)

    X_train_frame = make_frame_matrix(train_entries, key)
    X_test_frame = make_frame_matrix(test_entries, key)
    y_train_frame = make_frame_alpha(train_entries)
    y_test_frame = make_frame_alpha(test_entries)
    pred_train_frame, pred_test_frame = fit_ridge(X_train_frame, y_train_frame, X_test_frame, lam_frame)

    shuf = alpha_train.copy()
    rng.shuffle(shuf)
    _, pred_test_shuf = fit_ridge(X_train_sim, shuf, X_test_sim, lam_sim)

    return {
        "lambda_sim": float(lam_sim),
        "lambda_frame": float(lam_frame),
        "sim_train_r2": r2_score_np(alpha_train, pred_train_sim),
        "sim_test_r2": r2_score_np(alpha_test, pred_test_sim),
        "sim_test_mae": mae_np(alpha_test, pred_test_sim),
        "sim_test_r2_shuffled_label_control": r2_score_np(alpha_test, pred_test_shuf),
        "frame_train_r2": r2_score_np(y_train_frame, pred_train_frame),
        "frame_test_r2": r2_score_np(y_test_frame, pred_test_frame),
        "frame_test_mae": mae_np(y_test_frame, pred_test_frame),
    }


def fit_koopman(train_entries, test_entries, key, latent_dim, shuffle_train_time, seed):
    rng = np.random.RandomState(seed)
    train_frames = np.concatenate([e[key] for e in train_entries], axis=0).astype(np.float32)
    mean_frame = train_frames.mean(axis=0, keepdims=True)
    Xc = train_frames - mean_frame
    _, _, vh = np.linalg.svd(Xc, full_matrices=False)
    k = int(min(latent_dim, vh.shape[0], vh.shape[1]))
    W = vh[:k].T.astype(np.float32)

    def project(seq):
        return (seq - mean_frame) @ W

    def reconstruct(z):
        return z @ W.T + mean_frame

    z_t_parts = []
    z_n_parts = []
    for e in train_entries:
        seq = e[key]
        if shuffle_train_time:
            perm = np.arange(seq.shape[0])
            rng.shuffle(perm)
            seq = seq[perm]
        z = project(seq)
        z_t_parts.append(z[:-1])
        z_n_parts.append(z[1:])
    Z_train_t = np.concatenate(z_t_parts, axis=0)
    Z_train_next = np.concatenate(z_n_parts, axis=0)
    A = np.linalg.lstsq(Z_train_t, Z_train_next, rcond=None)[0].astype(np.float32)

    def eval_rollout(entries):
        cos_list = []
        mse_list = []
        for e in entries:
            seq = e[key]
            z_true = project(seq)
            z_roll = np.zeros_like(z_true)
            z_roll[0] = z_true[0]
            for t in range(1, z_true.shape[0]):
                z_roll[t] = z_roll[t - 1] @ A
            rec = reconstruct(z_roll)
            cos_list.append(cosine_mean_np(rec[1:], seq[1:]))
            mse_list.append(float(np.mean((rec[1:] - seq[1:]) ** 2)))
        return float(np.mean(cos_list)), float(np.std(cos_list)), float(np.mean(mse_list))

    train_cos, train_cos_std, train_mse = eval_rollout(train_entries)
    test_cos, test_cos_std, test_mse = eval_rollout(test_entries)
    return {
        "latent_dim": int(k),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(A)))),
        "rollout_cos_train_mean": train_cos,
        "rollout_cos_train_std": train_cos_std,
        "rollout_cos_test_mean": test_cos,
        "rollout_cos_test_std": test_cos_std,
        "rollout_mse_train_mean": train_mse,
        "rollout_mse_test_mean": test_mse,
    }


def run_cross_modal(train_entries, test_entries, pixel_key, lam_grid, folds, seed):
    rng = np.random.RandomState(seed)
    lam_p2e = tune_ridge_multitarget(train_entries, pixel_key, "embed", lam_grid, folds, rng)
    lam_e2p = tune_ridge_multitarget(train_entries, "embed", pixel_key, lam_grid, folds, rng)

    Xp_tr = make_frame_matrix(train_entries, pixel_key)
    Xp_te = make_frame_matrix(test_entries, pixel_key)
    Xe_tr = make_frame_matrix(train_entries, "embed")
    Xe_te = make_frame_matrix(test_entries, "embed")

    _, p2e_test = fit_ridge(Xp_tr, Xe_tr, Xp_te, lam_p2e)
    _, e2p_test = fit_ridge(Xe_tr, Xp_tr, Xe_te, lam_e2p)

    return {
        "lambda_pixel_to_embed": float(lam_p2e),
        "lambda_embed_to_pixel": float(lam_e2p),
        "pixel_to_embed_test_r2_total": total_variance_r2(Xe_te, p2e_test),
        "pixel_to_embed_test_cos": cosine_mean_np(Xe_te, p2e_test),
        "embed_to_pixel_test_r2_total": total_variance_r2(Xp_te, e2p_test),
        "embed_to_pixel_test_cos": cosine_mean_np(Xp_te, e2p_test),
    }


def mean_std_ci(values):
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.shape[0])
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    half_ci = (1.96 * std / math.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": float(mean - half_ci),
        "ci95_high": float(mean + half_ci),
    }


def main():
    parser = argparse.ArgumentParser(description="Stress-test whether embeddings behave like compressed pixels.")
    parser.add_argument("--sim-dir", type=Path, default=Path(__file__).resolve().parent / "simulations")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--ridge-grid", type=str, default="1e-5,1e-4,1e-3,1e-2,1e-1,1,10")
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--pixel-modes", type=str, default="rgb16,gray16,rgb32,crop_patch16")
    args = parser.parse_args()

    seeds = parse_list_arg(args.seeds, int)
    lam_grid = parse_list_arg(args.ridge_grid, float)
    pixel_modes = parse_list_arg(args.pixel_modes, str)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sim_dirs = sorted([d for d in args.sim_dir.iterdir() if d.is_dir()])
    entries = []

    for sim_dir in tqdm(sim_dirs, desc="Loading data"):
        params_path = sim_dir / "params.json"
        embed_path = sim_dir / "per_frame.npy"
        if not params_path.exists() or not embed_path.exists() or not (sim_dir / "frames").exists():
            continue

        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)
        embed = np.load(embed_path).astype(np.float32)
        if embed.ndim != 2 or embed.shape[0] < 4:
            continue

        frames = load_frames_float(sim_dir)
        feature_bank = {}
        for mode in pixel_modes:
            feature_bank[mode] = extract_pixel_features(frames, mode)

        T = embed.shape[0]
        for mode in pixel_modes:
            T = min(T, feature_bank[mode].shape[0])

        item = {
            "name": sim_dir.name,
            "alpha": float(params["alpha"]),
            "embed": embed[:T],
        }
        for mode in pixel_modes:
            item[f"pixel_{mode}"] = feature_bank[mode][:T]
        entries.append(item)

    if len(entries) < 10:
        raise ValueError("Not enough valid simulations. Run equation_sim.py and embed_sim.py first.")

    alpha_all = np.array([e["alpha"] for e in entries], dtype=np.float32)
    all_seed_results = []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        train_ids, test_ids = stratified_split_indices(alpha_all, args.test_ratio, rng, n_bins=5)
        train_entries = [entries[i] for i in train_ids]
        test_entries = [entries[i] for i in test_ids]

        seed_metrics = {
            "seed": int(seed),
            "split": {"n_train": len(train_entries), "n_test": len(test_entries)},
            "modalities": {},
            "cross_modal": {},
            "gaps": {},
        }

        keys = ["embed"] + [f"pixel_{m}" for m in pixel_modes]
        for key in keys:
            alpha_m = run_alpha_probes(train_entries, test_entries, key, lam_grid, args.cv_folds, seed)
            koop_m = fit_koopman(train_entries, test_entries, key, args.latent_dim, False, seed)
            koop_ctrl = fit_koopman(train_entries, test_entries, key, args.latent_dim, True, seed + 10000)
            seed_metrics["modalities"][key] = {
                "alpha_probe": alpha_m,
                "koopman": koop_m,
                "koopman_time_shuffled_control": koop_ctrl,
            }

        for mode in pixel_modes:
            key = f"pixel_{mode}"
            seed_metrics["cross_modal"][mode] = run_cross_modal(
                train_entries, test_entries, key, lam_grid, args.cv_folds, seed
            )

        embed_alpha = seed_metrics["modalities"]["embed"]["alpha_probe"]["sim_test_r2"]
        embed_koop = seed_metrics["modalities"]["embed"]["koopman"]["rollout_cos_test_mean"]
        pixel_alpha_vals = [seed_metrics["modalities"][f"pixel_{m}"]["alpha_probe"]["sim_test_r2"] for m in pixel_modes]
        pixel_koop_vals = [
            seed_metrics["modalities"][f"pixel_{m}"]["koopman"]["rollout_cos_test_mean"] for m in pixel_modes
        ]
        p2e_vals = [seed_metrics["cross_modal"][m]["pixel_to_embed_test_r2_total"] for m in pixel_modes]

        seed_metrics["gaps"]["embed_minus_best_pixel_alpha_sim_r2"] = float(embed_alpha - max(pixel_alpha_vals))
        seed_metrics["gaps"]["embed_minus_best_pixel_koop_rollout_cos"] = float(embed_koop - max(pixel_koop_vals))
        seed_metrics["gaps"]["best_pixel_to_embed_r2_total"] = float(max(p2e_vals))

        all_seed_results.append(seed_metrics)

    agg = {
        "embed_alpha_sim_test_r2": mean_std_ci(
            [r["modalities"]["embed"]["alpha_probe"]["sim_test_r2"] for r in all_seed_results]
        ),
        "embed_koop_rollout_cos_test": mean_std_ci(
            [r["modalities"]["embed"]["koopman"]["rollout_cos_test_mean"] for r in all_seed_results]
        ),
        "best_pixel_alpha_sim_test_r2": mean_std_ci(
            [
                max([r["modalities"][f"pixel_{m}"]["alpha_probe"]["sim_test_r2"] for m in pixel_modes])
                for r in all_seed_results
            ]
        ),
        "best_pixel_koop_rollout_cos_test": mean_std_ci(
            [
                max([r["modalities"][f"pixel_{m}"]["koopman"]["rollout_cos_test_mean"] for m in pixel_modes])
                for r in all_seed_results
            ]
        ),
        "gap_embed_minus_best_pixel_alpha_sim_r2": mean_std_ci(
            [r["gaps"]["embed_minus_best_pixel_alpha_sim_r2"] for r in all_seed_results]
        ),
        "gap_embed_minus_best_pixel_koop_rollout_cos": mean_std_ci(
            [r["gaps"]["embed_minus_best_pixel_koop_rollout_cos"] for r in all_seed_results]
        ),
        "best_pixel_to_embed_r2_total": mean_std_ci(
            [r["gaps"]["best_pixel_to_embed_r2_total"] for r in all_seed_results]
        ),
        "sign_consistency_alpha_gap_pos": float(
            np.mean([r["gaps"]["embed_minus_best_pixel_alpha_sim_r2"] > 0 for r in all_seed_results])
        ),
        "sign_consistency_koop_gap_pos": float(
            np.mean([r["gaps"]["embed_minus_best_pixel_koop_rollout_cos"] > 0 for r in all_seed_results])
        ),
    }

    alpha_gap_low = agg["gap_embed_minus_best_pixel_alpha_sim_r2"]["ci95_low"]
    koop_gap_low = agg["gap_embed_minus_best_pixel_koop_rollout_cos"]["ci95_low"]
    p2e_high = agg["best_pixel_to_embed_r2_total"]["ci95_high"]

    if alpha_gap_low > 0.0 and koop_gap_low > 0.0 and p2e_high < 0.8:
        verdict = "Consistent multi-seed evidence against a pixel-only explanation in this setup."
    elif agg["sign_consistency_alpha_gap_pos"] >= 0.8 and agg["sign_consistency_koop_gap_pos"] >= 0.8:
        verdict = "Suggestive evidence against a pixel-only explanation; uncertainty remains."
    else:
        verdict = "Inconclusive under current controls; expand baselines and dataset."

    report = {
        "setup": {
            "n_total": len(entries),
            "seeds": seeds,
            "test_ratio": float(args.test_ratio),
            "ridge_grid": lam_grid,
            "cv_folds": int(args.cv_folds),
            "latent_dim": int(args.latent_dim),
            "pixel_modes": pixel_modes,
        },
        "per_seed": all_seed_results,
        "aggregate": agg,
        "verdict": verdict,
    }

    out_json = args.out_dir / "pixel_falsification_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    lines = [
        "pixel-vs-embedding falsification report (multi-seed, tuned, stratified)",
        "=" * 78,
        f"sims={len(entries)} seeds={seeds} test_ratio={args.test_ratio}",
        f"pixel_modes={pixel_modes}",
        f"ridge_grid={lam_grid}",
        "",
        "aggregate (mean +/- std, 95% CI)",
        "-" * 78,
        (
            "embed alpha(sim) test R2: "
            f"{agg['embed_alpha_sim_test_r2']['mean']:.4f} +/- {agg['embed_alpha_sim_test_r2']['std']:.4f} "
            f"[{agg['embed_alpha_sim_test_r2']['ci95_low']:.4f}, {agg['embed_alpha_sim_test_r2']['ci95_high']:.4f}]"
        ),
        (
            "best pixel alpha(sim) test R2: "
            f"{agg['best_pixel_alpha_sim_test_r2']['mean']:.4f} +/- {agg['best_pixel_alpha_sim_test_r2']['std']:.4f} "
            f"[{agg['best_pixel_alpha_sim_test_r2']['ci95_low']:.4f}, {agg['best_pixel_alpha_sim_test_r2']['ci95_high']:.4f}]"
        ),
        (
            "gap alpha(sim) embed-bestpixel: "
            f"{agg['gap_embed_minus_best_pixel_alpha_sim_r2']['mean']:.4f} +/- "
            f"{agg['gap_embed_minus_best_pixel_alpha_sim_r2']['std']:.4f} "
            f"[{agg['gap_embed_minus_best_pixel_alpha_sim_r2']['ci95_low']:.4f}, "
            f"{agg['gap_embed_minus_best_pixel_alpha_sim_r2']['ci95_high']:.4f}]"
        ),
        (
            "embed koop rollout test cos: "
            f"{agg['embed_koop_rollout_cos_test']['mean']:.4f} +/- {agg['embed_koop_rollout_cos_test']['std']:.4f} "
            f"[{agg['embed_koop_rollout_cos_test']['ci95_low']:.4f}, {agg['embed_koop_rollout_cos_test']['ci95_high']:.4f}]"
        ),
        (
            "best pixel koop rollout test cos: "
            f"{agg['best_pixel_koop_rollout_cos_test']['mean']:.4f} +/- "
            f"{agg['best_pixel_koop_rollout_cos_test']['std']:.4f} "
            f"[{agg['best_pixel_koop_rollout_cos_test']['ci95_low']:.4f}, "
            f"{agg['best_pixel_koop_rollout_cos_test']['ci95_high']:.4f}]"
        ),
        (
            "gap koop embed-bestpixel: "
            f"{agg['gap_embed_minus_best_pixel_koop_rollout_cos']['mean']:.4f} +/- "
            f"{agg['gap_embed_minus_best_pixel_koop_rollout_cos']['std']:.4f} "
            f"[{agg['gap_embed_minus_best_pixel_koop_rollout_cos']['ci95_low']:.4f}, "
            f"{agg['gap_embed_minus_best_pixel_koop_rollout_cos']['ci95_high']:.4f}]"
        ),
        (
            "best pixel->embed total R2: "
            f"{agg['best_pixel_to_embed_r2_total']['mean']:.4f} +/- {agg['best_pixel_to_embed_r2_total']['std']:.4f} "
            f"[{agg['best_pixel_to_embed_r2_total']['ci95_low']:.4f}, {agg['best_pixel_to_embed_r2_total']['ci95_high']:.4f}]"
        ),
        (
            "sign consistency (alpha gap > 0, koop gap > 0): "
            f"{agg['sign_consistency_alpha_gap_pos']:.2f}, {agg['sign_consistency_koop_gap_pos']:.2f}"
        ),
        "",
        "verdict",
        "-" * 78,
        verdict,
    ]

    out_txt = args.out_dir / "pixel_falsification_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\nSaved: {out_txt}")
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
