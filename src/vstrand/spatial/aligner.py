"""Pillar 5 - 3D spatio-temporal grounding (STLLM-Aligner style).
Egocentric video tokens attend to allocentric 3D point-cloud tokens so local motion is
anchored in global coordinates."""
import torch, torch.nn as nn

class SpatioTemporalAligner(nn.Module):
    def __init__(self, vid_dim=1024, pc_dim=256, dim=128, heads=4):
        super().__init__()
        self.q, self.kv = nn.Linear(vid_dim, dim), nn.Linear(pc_dim, dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.pos = nn.Sequential(nn.Linear(3, dim), nn.ReLU(), nn.Linear(dim, dim))   # xyz -> embedding
        self.out = nn.Linear(dim, dim)

    def forward(self, video_tokens, pc_tokens, pc_xyz):
        # video_tokens [B,Nv,vid_dim]; pc_tokens [B,Np,pc_dim]; pc_xyz [B,Np,3]
        k = self.kv(pc_tokens) + self.pos(pc_xyz)
        fused, w = self.attn(self.q(video_tokens), k, k)
        return self.out(fused), w                     # w = which 3D points each token grounds to
