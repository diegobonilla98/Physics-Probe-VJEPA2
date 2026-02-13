import torch
import numpy as np
import imageio.v3 as iio
from transformers import AutoVideoProcessor, AutoModel
import torch.nn.functional as F

hf_repo = "facebook/vjepa2-vitl-fpc64-256"

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(hf_repo).to(device).eval()
processor = AutoVideoProcessor.from_pretrained(hf_repo)

num_frames = 16
video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/archery/-Qz25rXdMjE_000014_000024.mp4"

frames = np.stack([f for i, f in enumerate(iio.imiter(video_url)) if i < num_frames])
video = torch.from_numpy(frames).permute(0, 3, 1, 2)

inputs = processor(video, return_tensors="pt").to(device)

with torch.inference_mode():
    outputs = model(**inputs, skip_predictor=True)
    tokens = outputs.last_hidden_state

B, S, D = tokens.shape
P = (model.config.crop_size // model.config.patch_size) ** 2
T = S // P

tokens = tokens[:, :T * P].view(B, T, P, D)

per_frame = tokens.mean(dim=2)
global_emb = per_frame.mean(dim=1)

global_emb = F.normalize(global_emb, dim=-1)
per_frame = F.normalize(per_frame, dim=-1)

print(model.config.tubelet_size)  # 2
print("tokens:", tokens.shape)
print("per_frame:", per_frame.shape)
print("global:", global_emb.shape)
