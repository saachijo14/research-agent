import json
import queue
import threading
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent import cache, faults
from agent.pipeline import run_agent, run_baseline
from agent.trace import Tracer

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"

app = FastAPI(title="Evidence-Ledger Research Agent")

JOBS: dict[str, queue.Queue] = {}
RUN_LOCK = threading.Lock()  # the agent uses process-wide settings (fault, cache), so one run at a time

LAST_RESULT: dict = {}

class RunRequest(BaseModel):
    question: str = Field(min_length=5, max_length=500)
    mode: Literal["agent", "baseline"] = "agent"
    fault: str = "none"
    skeptic: bool = True
    replay: bool = False


class LiveTracer(Tracer):
    """Same JSONL trace as the CLI, but every event is also pushed to the browser."""

    def __init__(self, q: queue.Queue):
        super().__init__()
        self.q = q

    def log(self, event: str, **data):
        super().log(event, **data)
        self.q.put({"type": "event", "record": self.events[-1]})


def _worker(req: RunRequest, q: queue.Queue) -> None:
    try:
        faults.set_fault(req.fault if req.mode == "agent" else "none")
        cache.set_mode("replay" if req.replay else "auto")
        tracer = LiveTracer(q)
        if req.mode == "baseline":
            out = run_baseline(req.question, tracer=tracer)
        else:
            out = run_agent(req.question, tracer=tracer, use_skeptic=req.skeptic)
        data = out.to_dict()
        LAST_RESULT["data"] = data
        q.put({"type": "result", "data": data})
    except Exception as e:  # noqa: BLE001
        q.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
    finally:
        faults.set_fault("none")
        q.put(None)
        RUN_LOCK.release()


@app.post("/api/run")
def start_run(req: RunRequest):
    if req.fault not in faults.VALID_FAULTS:
        raise HTTPException(400, f"fault must be one of {sorted(faults.VALID_FAULTS)}")
    if not RUN_LOCK.acquire(blocking=False):
        raise HTTPException(409, "Another run is still in progress. Wait for it to finish.")
    job_id = uuid.uuid4().hex[:12]
    q: queue.Queue = queue.Queue()
    JOBS[job_id] = q
    threading.Thread(target=_worker, args=(req, q), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/stream/{job_id}")
def stream(job_id: str):
    q = JOBS.get(job_id)
    if q is None:
        raise HTTPException(404, "Unknown job")

    def gen():
        try:
            while True:
                try:
                    item = q.get(timeout=15)
                except queue.Empty:
                    yield ": keep-alive\n\n"
                    continue
                if item is None:
                    yield "event: done\ndata: {}\n\n"
                    break
                yield f"data: {json.dumps(item, default=str)}\n\n"
        finally:
            JOBS.pop(job_id, None)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/examples")
def examples():
    if not SAMPLES.exists():
        return []
    return [{"name": p.name, "label": p.stem.replace("_", " ")} for p in sorted(SAMPLES.glob("*.json"))]


@app.get("/api/example/{name}")
def example(name: str):
    p = SAMPLES / name
    if Path(name).name != name or p.suffix != ".json" or not p.exists():
        raise HTTPException(404, "Example not found")
    return json.loads(p.read_text(encoding="utf-8"))


class SaveRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60, pattern=r"^[A-Za-z0-9_\- ]+$")


@app.post("/api/save-last")
def save_last(req: SaveRequest):
    data = LAST_RESULT.get("data")
    if not data:
        raise HTTPException(400, "No run to save yet")
    SAMPLES.mkdir(exist_ok=True)
    fname = req.name.strip().replace(" ", "_") + ".json"
    (SAMPLES / fname).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return {"saved": fname}

@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")