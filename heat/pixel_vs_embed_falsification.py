import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


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


def total_variance_r2(y_true, y_pred):
    mu = y_true.mean(axis=0, keepdims=True)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - mu) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def load_frame_features(sim_dir, target_hw):
    frames_dir = sim_dir / "frames"
    frame_paths = sorted(
        p
        for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )
    if len(frame_paths) == 0:
        raise ValueError(f"No frames found in {frames_dir}")

    all_frames = np.stack([iio.imread(p) for p in frame_paths], axis=0)
    if all_frames.ndim == 3:
        all_frames = np.repeat(all_frames[..., None], 3, axis=-1)
    if all_frames.shape[-1] == 4:
        all_frames = all_frames[..., :3]

    x = torch.from_numpy(all_frames).float() / 255.0
    x = x.permute(0, 3, 1, 2).contiguous()
    x_small = F.interpolate(x, size=(target_hw, target_hw), mode="area")
    feats = x_small.permute(0, 2, 3, 1).contiguous().view(x_small.shape[0], -1)
    return feats.cpu().numpy().astype(np.float32)


def fit_koopman(train_entries, test_entries, key, latent_dim):
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

    Z_train_t = np.concatenate([project(e[key][:-1]) for e in train_entries], axis=0)
    Z_train_next = np.concatenate([project(e[key][1:]) for e in train_entries], axis=0)
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
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(A))))
    return {
        "latent_dim": int(k),
        "spectral_radius": spectral_radius,
        "rollout_cos_train_mean": train_cos,
        "rollout_cos_train_std": train_cos_std,
        "rollout_cos_test_mean": test_cos,
        "rollout_cos_test_std": test_cos_std,
        "rollout_mse_train_mean": train_mse,
        "rollout_mse_test_mean": test_mse,
    }


def main():
    parser = argparse.ArgumentParser(description="Falsify the 'embeddings are just raw pixels' hypothesis.")
    parser.add_argument("--sim-dir", type=Path, default=Path(__file__).resolve().parent / "simulations")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--pixel-grid", type=int, default=16)
    args = parser.parse_args()

    np.random.seed(args.seed)
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

        pixel = load_frame_features(sim_dir, args.pixel_grid)
        T = min(embed.shape[0], pixel.shape[0])
        entries.append(
            {
                "name": sim_dir.name,
                "alpha": float(params["alpha"]),
                "embed": embed[:T],
                "pixel": pixel[:T],
            }
        )

    if len(entries) < 6:
        raise ValueError("Not enough valid simulations. Run equation_sim.py and embed_sim.py first.")

    perm = np.random.permutation(len(entries))
    n_test = max(1, int(len(entries) * args.test_ratio))
    test_ids = set(perm[:n_test].tolist())
    train_entries = [entries[i] for i in range(len(entries)) if i not in test_ids]
    test_entries = [entries[i] for i in range(len(entries)) if i in test_ids]

    alpha_train = np.array([e["alpha"] for e in train_entries], dtype=np.float32)
    alpha_test = np.array([e["alpha"] for e in test_entries], dtype=np.float32)

    embed_train_sim = np.stack([e["embed"].mean(axis=0) for e in train_entries], axis=0).astype(np.float32)
    embed_test_sim = np.stack([e["embed"].mean(axis=0) for e in test_entries], axis=0).astype(np.float32)
    pixel_train_sim = np.stack([e["pixel"].mean(axis=0) for e in train_entries], axis=0).astype(np.float32)
    pixel_test_sim = np.stack([e["pixel"].mean(axis=0) for e in test_entries], axis=0).astype(np.float32)

    alpha_pred_embed_tr, alpha_pred_embed_te = fit_ridge(embed_train_sim, alpha_train, embed_test_sim, args.ridge)
    alpha_pred_pixel_tr, alpha_pred_pixel_te = fit_ridge(pixel_train_sim, alpha_train, pixel_test_sim, args.ridge)

    embed_train_frame = np.concatenate([e["embed"] for e in train_entries], axis=0)
    embed_test_frame = np.concatenate([e["embed"] for e in test_entries], axis=0)
    pixel_train_frame = np.concatenate([e["pixel"] for e in train_entries], axis=0)
    pixel_test_frame = np.concatenate([e["pixel"] for e in test_entries], axis=0)

    y_train_frame = np.concatenate([
        np.full((e["embed"].shape[0],), e["alpha"], dtype=np.float32) for e in train_entries
    ])
    y_test_frame = np.concatenate([
        np.full((e["embed"].shape[0],), e["alpha"], dtype=np.float32) for e in test_entries
    ])

    alpha_pred_embed_f_tr, alpha_pred_embed_f_te = fit_ridge(embed_train_frame, y_train_frame, embed_test_frame, args.ridge)
    alpha_pred_pixel_f_tr, alpha_pred_pixel_f_te = fit_ridge(pixel_train_frame, y_train_frame, pixel_test_frame, args.ridge)

    pixel_to_embed_train, pixel_to_embed_test = fit_ridge(pixel_train_frame, embed_train_frame, pixel_test_frame, args.ridge)
    embed_to_pixel_train, embed_to_pixel_test = fit_ridge(embed_train_frame, pixel_train_frame, embed_test_frame, args.ridge)

    koop_embed = fit_koopman(train_entries, test_entries, key="embed", latent_dim=args.latent_dim)
    koop_pixel = fit_koopman(train_entries, test_entries, key="pixel", latent_dim=args.latent_dim)

    metrics = {
        "setup": {
            "n_total": len(entries),
            "n_train": len(train_entries),
            "n_test": len(test_entries),
            "seed": int(args.seed),
            "test_ratio": float(args.test_ratio),
            "ridge_lambda": float(args.ridge),
            "latent_dim": int(args.latent_dim),
            "pixel_grid": int(args.pixel_grid),
            "pixel_dim": int(pixel_train_frame.shape[1]),
            "embed_dim": int(embed_train_frame.shape[1]),
        },
        "alpha_probe_sim": {
            "embed_train_r2": r2_score_np(alpha_train, alpha_pred_embed_tr),
            "embed_test_r2": r2_score_np(alpha_test, alpha_pred_embed_te),
            "embed_test_mae": mae_np(alpha_test, alpha_pred_embed_te),
            "pixel_train_r2": r2_score_np(alpha_train, alpha_pred_pixel_tr),
            "pixel_test_r2": r2_score_np(alpha_test, alpha_pred_pixel_te),
            "pixel_test_mae": mae_np(alpha_test, alpha_pred_pixel_te),
        },
        "alpha_probe_frame": {
            "embed_train_r2": r2_score_np(y_train_frame, alpha_pred_embed_f_tr),
            "embed_test_r2": r2_score_np(y_test_frame, alpha_pred_embed_f_te),
            "embed_test_mae": mae_np(y_test_frame, alpha_pred_embed_f_te),
            "pixel_train_r2": r2_score_np(y_train_frame, alpha_pred_pixel_f_tr),
            "pixel_test_r2": r2_score_np(y_test_frame, alpha_pred_pixel_f_te),
            "pixel_test_mae": mae_np(y_test_frame, alpha_pred_pixel_f_te),
        },
        "cross_modal_linear_predictability": {
            "pixel_to_embed_test_r2_total": total_variance_r2(embed_test_frame, pixel_to_embed_test),
            "pixel_to_embed_test_cos": cosine_mean_np(embed_test_frame, pixel_to_embed_test),
            "embed_to_pixel_test_r2_total": total_variance_r2(pixel_test_frame, embed_to_pixel_test),
            "embed_to_pixel_test_cos": cosine_mean_np(pixel_test_frame, embed_to_pixel_test),
        },
        "koopman_rollout": {
            "embed": koop_embed,
            "pixel": koop_pixel,
        },
    }

    emb_alpha_adv = metrics["alpha_probe_sim"]["embed_test_r2"] - metrics["alpha_probe_sim"]["pixel_test_r2"]
    emb_dyn_adv = (
        metrics["koopman_rollout"]["embed"]["rollout_cos_test_mean"]
        - metrics["koopman_rollout"]["pixel"]["rollout_cos_test_mean"]
    )
    pix_to_emb_r2 = metrics["cross_modal_linear_predictability"]["pixel_to_embed_test_r2_total"]

    if emb_alpha_adv > 0.1 and emb_dyn_adv > 0.05 and pix_to_emb_r2 < 0.95:
        verdict = "Evidence against catastrophic pixel-only explanation."
    elif pix_to_emb_r2 > 0.98 and abs(emb_dyn_adv) < 0.02 and abs(emb_alpha_adv) < 0.05:
        verdict = "High risk that embeddings are mostly linearized pixels."
    else:
        verdict = "Mixed evidence; run extra controls (color-map perturbation, spatial shuffles, multi-seed)."

    metrics["verdict"] = {
        "embedding_alpha_advantage_r2": float(emb_alpha_adv),
        "embedding_dynamics_advantage_cos": float(emb_dyn_adv),
        "pixel_to_embedding_linear_r2": float(pix_to_emb_r2),
        "text": verdict,
    }

    out_json = args.out_dir / "pixel_falsification_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    lines = [
        "pixel-vs-embedding falsification report",
        "=" * 68,
        f"sims: total={len(entries)} train={len(train_entries)} test={len(test_entries)}",
        f"pixel_features={args.pixel_grid}x{args.pixel_grid}x3 ({pixel_train_frame.shape[1]} dims)",
        f"embedding_dim={embed_train_frame.shape[1]} latent_dim={args.latent_dim}",
        "",
        "alpha probe (simulation-mean)",
        "-" * 68,
        f"embed: test_r2={metrics['alpha_probe_sim']['embed_test_r2']:.4f} test_mae={metrics['alpha_probe_sim']['embed_test_mae']:.4f}",
        f"pixel: test_r2={metrics['alpha_probe_sim']['pixel_test_r2']:.4f} test_mae={metrics['alpha_probe_sim']['pixel_test_mae']:.4f}",
        f"embed_minus_pixel_r2={emb_alpha_adv:.4f}",
        "",
        "alpha probe (frame-level)",
        "-" * 68,
        f"embed: test_r2={metrics['alpha_probe_frame']['embed_test_r2']:.4f} test_mae={metrics['alpha_probe_frame']['embed_test_mae']:.4f}",
        f"pixel: test_r2={metrics['alpha_probe_frame']['pixel_test_r2']:.4f} test_mae={metrics['alpha_probe_frame']['pixel_test_mae']:.4f}",
        "",
        "cross-modal linear predictability",
        "-" * 68,
        f"pixel->embed: test_r2_total={metrics['cross_modal_linear_predictability']['pixel_to_embed_test_r2_total']:.4f} cos={metrics['cross_modal_linear_predictability']['pixel_to_embed_test_cos']:.4f}",
        f"embed->pixel: test_r2_total={metrics['cross_modal_linear_predictability']['embed_to_pixel_test_r2_total']:.4f} cos={metrics['cross_modal_linear_predictability']['embed_to_pixel_test_cos']:.4f}",
        "",
        "koopman rollout (same latent dim)",
        "-" * 68,
        f"embed: test_cos={metrics['koopman_rollout']['embed']['rollout_cos_test_mean']:.4f} +/- {metrics['koopman_rollout']['embed']['rollout_cos_test_std']:.4f}",
        f"pixel: test_cos={metrics['koopman_rollout']['pixel']['rollout_cos_test_mean']:.4f} +/- {metrics['koopman_rollout']['pixel']['rollout_cos_test_std']:.4f}",
        f"embed_minus_pixel_cos={emb_dyn_adv:.4f}",
        "",
        "verdict",
        "-" * 68,
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
