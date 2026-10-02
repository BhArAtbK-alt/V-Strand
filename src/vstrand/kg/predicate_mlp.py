"""Pillar 4 - Neural knowledge-graph extraction.
Learnable entity vectors + video encoding -> predicate MLPs -> P(unary) and P(binary) for
<subject, predicate, object> triples."""
import torch, torch.nn as nn

class PredicateKG(nn.Module):
    def __init__(self, n_entities=8, entity_dim=64, video_dim=64, n_unary=4, n_binary=6):
        super().__init__()
        self.entities = nn.Parameter(torch.randn(n_entities, entity_dim) * 0.1)   # learnable slots
        self.unary = nn.ModuleList([self._mlp(entity_dim + video_dim) for _ in range(n_unary)])
        self.binary = nn.ModuleList([self._mlp(2 * entity_dim + video_dim) for _ in range(n_binary)])

    @staticmethod
    def _mlp(i):
        return nn.Sequential(nn.Linear(i, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, video):                       # video [B, video_dim]
        B, E = video.shape[0], self.entities.shape[0]
        ent = self.entities.unsqueeze(0).expand(B, E, -1)
        v = video.unsqueeze(1).expand(B, E, -1)
        u = torch.cat([ent, v], -1)                                              # [B,E,*]
        unary = torch.cat([torch.sigmoid(m(u)) for m in self.unary], -1)          # [B,E,n_unary]
        s = ent.unsqueeze(2).expand(B, E, E, -1); o = ent.unsqueeze(1).expand(B, E, E, -1)
        vv = video[:, None, None, :].expand(B, E, E, -1)
        b = torch.cat([s, o, vv], -1)
        binary = torch.cat([torch.sigmoid(m(b)) for m in self.binary], -1)        # [B,E,E,n_binary]
        return unary, binary
