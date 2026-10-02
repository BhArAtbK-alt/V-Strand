"""Pillar 1 - Alignment-Before-Projection.
Image and video features come from a PRE-ALIGNED encoder (LanguageBind), so they already
live in one language-grounded space. ONE shared projector (not one per modality) maps them
into the LLM embedding space."""
import torch, torch.nn as nn

class SharedProjector(nn.Module):
    def __init__(self, vis_dim=1024, llm_dim=4096):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(vis_dim, llm_dim), nn.GELU(), nn.Linear(llm_dim, llm_dim))

    def forward(self, image_feats=None, video_feats=None):
        parts = [x for x in (image_feats, video_feats) if x is not None]   # [B,N,D] each
        return self.net(torch.cat(parts, dim=1))
