import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
base_dir = script_dir / "simulations"

nx = 256
ny = 256
dx = 1.0
dt = 0.1
steps = 64
substeps = 10

n_sims = 50
seed = 42

rng = np.random.RandomState(seed)

x = np.arange(nx)
y = np.arange(ny)
X, Y = np.meshgrid(x, y, indexing="ij")

for sim_idx in tqdm(range(n_sims), desc="Simulations"):
    alpha = rng.uniform(0.5, 5.0)
    sigma = rng.uniform(5.0, 20.0)
    cx = rng.randint(nx // 4, 3 * nx // 4)
    cy = rng.randint(ny // 4, 3 * ny // 4)

    sim_dir = base_dir / f"sim_{sim_idx:04d}"
    frames_dir = sim_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    params = {
        "alpha": float(alpha),
        "sigma": float(sigma),
        "cx": int(cx),
        "cy": int(cy),
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dt": dt,
        "steps": steps,
        "substeps": substeps,
    }
    with open(sim_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    u = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma ** 2))
    coef = alpha * dt / (dx ** 2)

    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])

    for frame in tqdm(range(steps), desc=f"sim_{sim_idx:04d}", leave=False):
        for _ in range(substeps):
            laplacian = (
                u[2:, 1:-1]
                + u[:-2, 1:-1]
                + u[1:-1, 2:]
                + u[1:-1, :-2]
                - 4 * u[1:-1, 1:-1]
            )
            u[1:-1, 1:-1] += coef * laplacian

        ax.cla()
        ax.imshow(u, cmap="inferno", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(
            frames_dir / f"frame_{frame:04d}.png",
            bbox_inches="tight",
            pad_inches=0,
        )

    plt.close(fig)

print("Done.")
