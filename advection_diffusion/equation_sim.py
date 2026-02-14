import json
import shutil
from pathlib import Path

import imageio.v3 as iio
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm

script_dir = Path(__file__).resolve().parent
base_dir = script_dir / "simulations"

nx = 128
ny = 128
steps = 64
n_pairs = 60  # total simulations = 2 * n_pairs
seed = 123
reset_existing = True

# Advection is implemented as exact periodic integer shifts.
velocity_choices = [-2, -1, 1, 2]

# Diffusion is explicit Euler with periodic laplacian:
# u_{t+1} = u_t + alpha * dt * lap(u_t)
# Keep alpha * dt <= 0.25 for stability in 2D.
diff_dt = 0.08
alpha_candidates = np.linspace(0.01, 0.22, 28, dtype=np.float32)

rng = np.random.RandomState(seed)
cmap = cm.get_cmap("inferno")

x = np.arange(nx, dtype=np.float32)
y = np.arange(ny, dtype=np.float32)
X, Y = np.meshgrid(x, y, indexing="ij")


def periodic_laplacian(u):
    return (
        np.roll(u, 1, axis=0)
        + np.roll(u, -1, axis=0)
        + np.roll(u, 1, axis=1)
        + np.roll(u, -1, axis=1)
        - 4.0 * u
    )


def gradient_smoothness(u):
    gx = np.roll(u, -1, axis=0) - u
    gy = np.roll(u, -1, axis=1) - u
    return float(np.mean(np.sqrt(gx * gx + gy * gy)))


def smoothness_profile(clip):
    return np.array([gradient_smoothness(frame) for frame in clip], dtype=np.float32)


def make_initial_blob_field(local_rng):
    n_blobs = int(local_rng.randint(1, 4))
    u0 = np.zeros((nx, ny), dtype=np.float32)
    for _ in range(n_blobs):
        amp = float(local_rng.uniform(0.5, 1.2))
        sigma = float(local_rng.uniform(4.0, 14.0))
        cx = float(local_rng.uniform(0.2 * nx, 0.8 * nx))
        cy = float(local_rng.uniform(0.2 * ny, 0.8 * ny))
        blob = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma * sigma))
        u0 += amp * blob.astype(np.float32)
    u0 = u0 / (float(u0.max()) + 1e-8)
    return np.clip(u0, 0.0, 1.0).astype(np.float32)


def run_advection(u0, vx, vy):
    clip = np.zeros((steps, nx, ny), dtype=np.float32)
    u = u0.copy()
    for t in range(steps):
        clip[t] = u
        u = np.roll(u, shift=vx, axis=0)
        u = np.roll(u, shift=vy, axis=1)
    return clip


def run_diffusion(u0, alpha):
    coef = float(alpha) * diff_dt
    if coef > 0.249:
        raise ValueError(f"Unstable diffusion coefficient: alpha*dt={coef:.4f}")
    clip = np.zeros((steps, nx, ny), dtype=np.float32)
    u = u0.copy()
    for t in range(steps):
        clip[t] = u
        u = u + coef * periodic_laplacian(u)
        u = np.clip(u, 0.0, 1.0)
    return clip


def choose_diffusion_alpha(u0, target_profile):
    best_alpha = float(alpha_candidates[0])
    best_clip = None
    best_profile = None
    best_loss = float("inf")
    for alpha in alpha_candidates:
        clip = run_diffusion(u0, float(alpha))
        profile = smoothness_profile(clip)
        loss = float(np.mean((profile - target_profile) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_alpha = float(alpha)
            best_clip = clip
            best_profile = profile
    return best_alpha, best_clip, best_profile, best_loss


def save_clip_frames(clip, out_dir):
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for t in range(clip.shape[0]):
        rgb = cmap(np.clip(clip[t], 0.0, 1.0))[..., :3]
        img = np.clip(255.0 * rgb, 0.0, 255.0).astype(np.uint8)
        iio.imwrite(frames_dir / f"frame_{t:04d}.png", img)


def write_sim(sim_dir, params, clip):
    sim_dir.mkdir(parents=True, exist_ok=True)
    with open(sim_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    save_clip_frames(clip, sim_dir)


if reset_existing and base_dir.exists():
    for old_dir in base_dir.glob("sim_*"):
        if old_dir.is_dir():
            shutil.rmtree(old_dir)

base_dir.mkdir(parents=True, exist_ok=True)

manifest = []
sim_index = 0
for pair_id in tqdm(range(n_pairs), desc="Advection-vs-diffusion pairs"):
    u0 = make_initial_blob_field(rng)

    vx = int(rng.choice(velocity_choices))
    vy = int(rng.choice(velocity_choices))
    while vx == 0 and vy == 0:
        vx = int(rng.choice(velocity_choices))
        vy = int(rng.choice(velocity_choices))

    adv_clip = run_advection(u0, vx, vy)
    adv_profile = smoothness_profile(adv_clip)

    alpha, diff_clip, diff_profile, match_loss = choose_diffusion_alpha(u0, adv_profile)

    adv_params = {
        "equation": "advection",
        "pair_id": int(pair_id),
        "vx": int(vx),
        "vy": int(vy),
        "alpha": 0.0,
        "nx": int(nx),
        "ny": int(ny),
        "steps": int(steps),
        "diff_dt": float(diff_dt),
        "smoothness_mean": float(np.mean(adv_profile)),
        "match_target": "advection_profile",
    }
    adv_dir = base_dir / f"sim_{sim_index:04d}"
    write_sim(adv_dir, adv_params, adv_clip)
    manifest.append({"sim": adv_dir.name, **adv_params})
    sim_index += 1

    diff_params = {
        "equation": "diffusion",
        "pair_id": int(pair_id),
        "vx": 0,
        "vy": 0,
        "alpha": float(alpha),
        "nx": int(nx),
        "ny": int(ny),
        "steps": int(steps),
        "diff_dt": float(diff_dt),
        "smoothness_mean": float(np.mean(diff_profile)),
        "match_target": "advection_profile",
        "match_loss": float(match_loss),
    }
    diff_dir = base_dir / f"sim_{sim_index:04d}"
    write_sim(diff_dir, diff_params, diff_clip)
    manifest.append({"sim": diff_dir.name, **diff_params})
    sim_index += 1

with open(base_dir / "simulation_manifest.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "experiment": "advection_vs_diffusion",
            "n_pairs": int(n_pairs),
            "n_total": int(2 * n_pairs),
            "seed": int(seed),
            "steps": int(steps),
            "nx": int(nx),
            "ny": int(ny),
            "velocity_choices": velocity_choices,
            "diffusion_dt": float(diff_dt),
            "alpha_candidates": [float(a) for a in alpha_candidates],
            "entries": manifest,
        },
        f,
        indent=2,
    )

print(f"Done. Generated {2 * n_pairs} simulations in {base_dir}")
