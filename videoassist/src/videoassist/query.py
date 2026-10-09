"""Query understanding: temporal relation, anchor event, expected answer type."""
import re
from .schema import QueryPlan
REL = re.compile(r"^(?P<target>.*?)\b(?P<rel>before|after|during|while)\b\s+(?:the technician\s+|he\s+|she\s+)?(?P<anchor>.+?)\??$", re.I)

def parse(q):
    m, rel, anchor, target = REL.match(q.strip()), None, None, q
    if m:
        rel = "during" if m["rel"].lower() == "while" else m["rel"].lower()
        anchor, target = m["anchor"], (m["target"].strip() or q)
    ql = q.lower()
    wants = ("code" if "code" in ql else "time" if ql.startswith("when")
             else "yesno" if re.match(r"(did|does|is|was|were|are)\b", ql) else "what")
    return QueryPlan(q, target, rel, anchor, wants)
