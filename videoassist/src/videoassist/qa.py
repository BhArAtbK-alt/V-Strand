"""Evidence-grounded answering. The answerer only ever sees retrieved evidence."""
from .verify import find_code

def extractive(plan, ev, strict):
    if plan.wants == "code":
        code = find_code(ev)
        if code: return f"Error code {code} is shown on the device."
        if strict: return None
    vis = [e for e in ev if e.modality == "visual"]; sp = [e for e in ev if e.modality == "speech"]
    ocr = [e for e in ev if e.modality == "ocr"]
    parts = ([vis[0].text.rstrip(".")] if vis else []) + ([f'Technician says: "{sp[0].text}"'] if sp else []) \
            + ([f"On-screen text: {ocr[0].text}"] if ocr else [])
    return ". ".join(parts) or None

PROMPT = ("Answer ONLY from the evidence below. If it is insufficient, reply exactly INSUFFICIENT.\n"
          "Question: {q}\nEvidence:\n{ev}\nAnswer:")

def llm_answer(plan, ev, generate):
    """generate: callable(str)->str wrapping a pretrained VLM/LLM. No general-knowledge answering allowed."""
    block = "\n".join(f"[{e.t0:.0f}-{e.t1:.0f}s {e.modality}] {e.text}" for e in ev)
    out = generate(PROMPT.format(q=plan.raw, ev=block)).strip()
    return None if out.upper().startswith("INSUFFICIENT") else out
