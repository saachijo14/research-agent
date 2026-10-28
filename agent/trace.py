import json
import time
from pathlib import Path


class Tracer:
    """Writes one JSON line per event to traces/<timestamp>.jsonl"""

    def __init__(self):
        Path("traces").mkdir(exist_ok=True)
        self.path = Path("traces") / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        self.events: list[dict] = []

    def log(self, event: str, **data):
        record = {"t": round(time.time(), 3), "event": event, **data}
        self.events.append(record)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")