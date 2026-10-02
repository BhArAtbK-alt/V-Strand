"""Background rules as differentiable penalties (Lukasiewicz-style implication):
   forall x,y  fold(x,y) => change(x,y)   ==>   penalty = relu(P(fold) - P(change))"""
import torch

RULES = [(0, 1)]   # (antecedent binary-predicate idx, consequent idx): fold -> change

def ontology_loss(binary_probs):
    loss = 0.0
    for a, c in RULES:
        loss = loss + torch.relu(binary_probs[..., a] - binary_probs[..., c]).mean()
    return loss
