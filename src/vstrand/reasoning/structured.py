"""Pillar 3 - Post-retrieval Structured Reasoning.
Decompose query -> verify each retrieved clip against each sub-query -> drop hard negatives."""
def structured_reasoning(query, candidate_clips, descriptions, lvlm, min_pass_ratio=1.0):
    subs = lvlm.decompose(query)
    kept, trace = [], []
    for cid in candidate_clips:
        votes = [lvlm.verify(descriptions[cid], s) for s in subs]
        ok = sum(votes) / len(votes) >= min_pass_ratio
        trace.append({"clip": cid, "votes": votes, "pass": ok})
        if ok:
            kept.append(cid)
    return kept, subs, trace
