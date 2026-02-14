import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoVideoProcessor


hf_repo = "facebook/vjepa2-vitl-fpc64-256"
script_dir = Path(__file__).resolve().parent
base_dir = script_dir / "simulations"
overwrite_embeddings = False

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(hf_repo).to(device).eval()
processor = AutoVideoProcessor.from_pretrained(hf_repo)

sim_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("sim_")])
print(f"Found {len(sim_dirs)} simulations")

for sim_dir in tqdm(sim_dirs, desc="Embedding"):
    per_frame_raw_path = sim_dir / "per_frame.npy"
    per_frame_norm_path = sim_dir / "per_frame_norm.npy"
    tokens_spatial_path = sim_dir / "tokens_spatial_fp16.npy"
    meta_path = sim_dir / "embed_meta.json"
    if (
        not overwrite_embeddings
        and per_frame_raw_path.exists()
        and per_frame_norm_path.exists()
        and tokens_spatial_path.exists()
        and meta_path.exists()
    ):
        tqdm.write(f"  [{sim_dir.name}] already embedded, skipping")
        continue

    frames_dir = sim_dir / "frames"
    if not frames_dir.exists():
        tqdm.write(f"  [{sim_dir.name}] no frames folder, skipping")
        continue

    with open(sim_dir / "params.json", "r", encoding="utf-8") as f:
        params = json.load(f)

    frame_paths = sorted(
        p
        for p in frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    )
    if len(frame_paths) == 0:
        tqdm.write(f"  [{sim_dir.name}] no frames found, skipping")
        continue

    all_frames = np.stack([iio.imread(p) for p in frame_paths], axis=0)
    if all_frames.shape[-1] == 4:
        all_frames = all_frames[..., :3]
    video = torch.from_numpy(all_frames).permute(0, 3, 1, 2)

    clip_frames = params.get("steps", video.shape[0])
    per_frame_raw_all = []
    per_frame_norm_all = []
    tokens_spatial_all = []
    token_grid_h = None
    token_grid_w = None
    token_dim = None

    for start in range(0, video.shape[0], clip_frames):
        end = min(start + clip_frames, video.shape[0])
        clip_video = video[start:end]
        inputs = processor(clip_video, return_tensors="pt").to(device)

        with torch.inference_mode():
            outputs = model(**inputs, skip_predictor=True)
            tokens = outputs.last_hidden_state

        B, S, D = tokens.shape
        P = (model.config.crop_size // model.config.patch_size) ** 2
        T = S // P
        tokens = tokens[:, : T * P].view(B, T, P, D)
        per_frame_raw = tokens.mean(dim=2)
        per_frame_norm = F.normalize(per_frame_raw, dim=-1)

        side = int(np.sqrt(P))
        if side * side != P:
            raise ValueError(f"Token grid is not square for {sim_dir.name}. P={P}")

        tokens_spatial = tokens.view(B, T, side, side, D)
        per_frame_raw_all.append(per_frame_raw.squeeze(0).cpu().numpy().astype(np.float32))
        per_frame_norm_all.append(per_frame_norm.squeeze(0).cpu().numpy().astype(np.float32))
        tokens_spatial_all.append(tokens_spatial.squeeze(0).cpu().numpy().astype(np.float16))
        token_grid_h = side
        token_grid_w = side
        token_dim = D

    per_frame_raw_np = np.concatenate(per_frame_raw_all, axis=0)
    per_frame_norm_np = np.concatenate(per_frame_norm_all, axis=0)
    tokens_spatial_np = np.concatenate(tokens_spatial_all, axis=0)

    np.save(per_frame_raw_path, per_frame_raw_np)
    np.save(per_frame_norm_path, per_frame_norm_np)
    np.save(tokens_spatial_path, tokens_spatial_np)

    embed_meta = {
        "hf_repo": hf_repo,
        "device": device,
        "temporal_tokens": int(tokens_spatial_np.shape[0]),
        "token_grid_h": int(token_grid_h),
        "token_grid_w": int(token_grid_w),
        "token_dim": int(token_dim),
        "per_frame_dim": int(per_frame_raw_np.shape[1]),
        "saved_files": [
            "per_frame.npy",
            "per_frame_norm.npy",
            "tokens_spatial_fp16.npy",
        ],
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(embed_meta, f, indent=2)

print("Done.")
