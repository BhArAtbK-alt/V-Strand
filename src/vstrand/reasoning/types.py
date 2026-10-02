from dataclasses import dataclass, field

@dataclass
class SubQuery:
    text: str
    kind: str = "binary"                            # binary | numeric
    keywords: list = field(default_factory=list)    # used only by MockLVLM
