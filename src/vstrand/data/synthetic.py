"""Tiny synthetic 'long video' so the full pipeline runs with no downloads.
Each clip = caption (what the vision LVLM would see) + transcript (ASR) + timestamps.
Replace with real clips from data/raw via clip_splitter.py."""
from dataclasses import dataclass

@dataclass
class Clip:
    clip_id: int
    start: float
    end: float
    caption: str                     # stands in for visual content
    transcript: str = ""             # timestamped speech
    frames: object = None            # real runs: array/tensor of sampled frames

def make_demo_video() -> list:
    rows = [
        ("A person walks into the office carrying a backpack.", "Morning everyone."),
        ("The person sits at the desk. The laptop is closed on the desk.", "Let me get set up."),
        ("The person opens the laptop and the screen turns on.", "Okay, booting up."),
        ("The person types on the laptop while a coffee mug sits beside it.", "Checking my email."),
        ("The person closes the laptop and picks up the coffee mug.", "Time for a break."),
        ("The person walks to the kitchen and pours coffee into the mug.", "Need a refill."),
        ("The person returns, sits down and the laptop is closed on the desk.", "Back at it."),
    ]
    return [Clip(i, i * 10.0, (i + 1) * 10.0, c, t) for i, (c, t) in enumerate(rows)]

DEMO_QUERIES = ["Did I open the laptop?", "Did I pour coffee into the mug?"]
