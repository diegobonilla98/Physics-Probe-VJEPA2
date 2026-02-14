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
seed = 321
reset_existing = True

# Damped wave:
# u_tt + gamma*u_t = c^2 * Delta u
# Explicit update with dt and periodic laplacian.
wave_dt = 0.2
c_range = (0.30, 0.90)
gamma_range = (0.01, 0.08)

# Diffusion baseline for matched initial condition:
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


def run_damped_wave(u0, c_val, gamma_val):
    cdt2 = (float(c_val) * wave_dt) ** 2
    damp = float(gamma_val) * wave_dt
    clip = np.zeros((steps, nx, ny), dtype=np.float32)

    # Zero initial velocity.
    u_prev = u0.copy()
    u = u0 + 0.5 * cdt2 * periodic_laplacian(u0)
    clip[0] = u0
    for t in range(1, steps):
        clip[t] = u
        lap = periodic_laplacian(u)
        u_next = (2.0 - damp) * u - (1.0 - damp) * u_prev + cdt2 * lap
        u_prev, u = u, u_next
        if not np.isfinite(u).all():
            raise FloatingPointError("Wave simulation became non-finite")
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


def save_clip_frames(clip, out_dir, render_scale):
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for t in range(clip.shape[0]):
        # Use a shared signed mapping across both equations in a pair.
        mapped = 0.5 + 0.5 * (clip[t] / (render_scale + 1e-8))
        rgb = cmap(np.clip(mapped, 0.0, 1.0))[..., :3]
        img = np.clip(255.0 * rgb, 0.0, 255.0).astype(np.uint8)
        iio.imwrite(frames_dir / f"frame_{t:04d}.png", img)


def write_sim(sim_dir, params, clip, render_scale):
    sim_dir.mkdir(parents=True, exist_ok=True)
    with open(sim_dir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    save_clip_frames(clip, sim_dir, render_scale)


if reset_existing and base_dir.exists():
    for old_dir in base_dir.glob("sim_*"):
        if old_dir.is_dir():
            shutil.rmtree(old_dir)

base_dir.mkdir(parents=True, exist_ok=True)

manifest = []
sim_index = 0
for pair_id in tqdm(range(n_pairs), desc="Wave-vs-diffusion pairs"):
    u0 = make_initial_blob_field(rng)
    c_val = float(rng.uniform(*c_range))
    gamma_val = float(rng.uniform(*gamma_range))

    wave_clip = run_damped_wave(u0, c_val, gamma_val)
    wave_profile = smoothness_profile(wave_clip)

    alpha, diff_clip, diff_profile, match_loss = choose_diffusion_alpha(u0, wave_profile)
    render_scale = float(max(np.max(np.abs(wave_clip)), np.max(np.abs(diff_clip)), 1.0))

    wave_params = {
        "equation": "wave",
        "pair_id": int(pair_id),
        "c": float(c_val),
        "gamma": float(gamma_val),
        "alpha": 0.0,
        "nx": int(nx),
        "ny": int(ny),
        "steps": int(steps),
        "wave_dt": float(wave_dt),
        "render_scale": float(render_scale),
        "smoothness_mean": float(np.mean(wave_profile)),
        "match_target": "wave_profile",
    }
    wave_dir = base_dir / f"sim_{sim_index:04d}"
    write_sim(wave_dir, wave_params, wave_clip, render_scale)
    manifest.append({"sim": wave_dir.name, **wave_params})
    sim_index += 1

    diff_params = {
        "equation": "diffusion",
        "pair_id": int(pair_id),
        "c": 0.0,
        "gamma": 0.0,
        "alpha": float(alpha),
        "nx": int(nx),
        "ny": int(ny),
        "steps": int(steps),
        "diff_dt": float(diff_dt),
        "render_scale": float(render_scale),
        "smoothness_mean": float(np.mean(diff_profile)),
        "match_target": "wave_profile",
        "match_loss": float(match_loss),
    }
    diff_dir = base_dir / f"sim_{sim_index:04d}"
    write_sim(diff_dir, diff_params, diff_clip, render_scale)
    manifest.append({"sim": diff_dir.name, **diff_params})
    sim_index += 1

with open(base_dir / "simulation_manifest.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "experiment": "wave_vs_diffusion",
            "n_pairs": int(n_pairs),
            "n_total": int(2 * n_pairs),
            "seed": int(seed),
            "steps": int(steps),
            "nx": int(nx),
            "ny": int(ny),
            "wave_dt": float(wave_dt),
            "c_range": [float(c_range[0]), float(c_range[1])],
            "gamma_range": [float(gamma_range[0]), float(gamma_range[1])],
            "diffusion_dt": float(diff_dt),
            "alpha_candidates": [float(a) for a in alpha_candidates],
            "entries": manifest,
        },
        f,
        indent=2,
    )

print(f"Done. Generated {2 * n_pairs} simulations in {base_dir}")
