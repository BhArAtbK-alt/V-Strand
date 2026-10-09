"""Evidence verification: abstain unless the query's content is covered by the grounded evidence."""
import re
from .text import tokens, stem, STOP
COVER_STOP = STOP | {"code", "appears", "appear", "shown", "displayed", "display", "screen", "router",
                     "happens", "happen", "time", "device", "equipment", "customer"}
CODE = re.compile(r"\b(?:E|ERR)-?\d{3,5}\b")

def coverage(text, evs):
    q = {stem(w) for w in tokens(text) if w not in COVER_STOP}
    if not q:
        return 1.0
    have = {stem(w) for e in evs for w in tokens(e.text)}
    return len(q & have) / len(q)

def find_code(evs):
    for e in evs:
        if e.modality == "ocr" and (m := CODE.search(e.text)):
            return m.group(0)
    return None
