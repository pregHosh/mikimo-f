"""Machine-readable run record: ``mikimo_result.json`` (schema ``mikimo_result/v1``).

One record is started when a mode resolves its output directory (``helper.preprocess_data_mkm``)
and is written there when the mode handler returns or fails (``__main__._get_mode_handler``).
Every run of ``mkm``, ``vp`` and ``cond`` writes one, also when it fails after the output
directory exists. ``mkm_output.json`` is still written with its old keys for older readers.
"""
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "mikimo_result/v1"
VERSION = "2.0.1"
FILENAME = "mikimo_result.json"

_current = None

_KINDS = {
    ".json": "json",
    ".csv": "csv",
    ".h5": "hdf5",
    ".png": "png",
}


def _clean(x):
    """JSON-safe copy: numpy scalars/arrays to python, NaN and inf to None."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "tolist") and not isinstance(x, (str, bytes)):
        return _clean(x.tolist())
    if isinstance(x, bytes):
        return x.decode("utf-8", "replace")
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, (bool, int, str)) or x is None:
        return x
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return str(x)


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RunRecord:
    def __init__(self, mode, outdir):
        self.outdir = Path(outdir)
        self.data = {
            "schema": SCHEMA,
            "version": VERSION,
            "mode": mode,
            "status": "running",
            "error": None,
            "inputs": {"files": []},
            "validation": {"clear": None, "issues": []},
            "species": [],
            "pathways": [],
            "mkm": None,
            "vp": None,
            "cond": None,
            "artifacts": [],
            "warnings": [],
        }

    def write(self):
        arts = []
        for p in sorted(self.outdir.rglob("*")):
            if p.is_file() and p.name != FILENAME:
                arts.append(
                    {
                        "path": str(p.relative_to(self.outdir)),
                        "kind": _KINDS.get(p.suffix.lower(), "other"),
                    }
                )
        self.data["artifacts"] = arts
        with open(self.outdir / FILENAME, "w", encoding="utf-8") as f:
            json.dump(_clean(self.data), f, indent=2)


def start(mode, record_dir, **inputs):
    """Start the record of this run, written to record_dir (replaces any record left from an
    earlier in-process run)."""
    global _current
    _current = RunRecord(mode, record_dir)
    _current.data["inputs"].update(inputs)
    return _current


def current():
    return _current


def update(section, **fields):
    """Merge fields into a section of the record (inputs, validation, mkm, vp, cond, ...)."""
    if _current is None:
        return
    d = _current.data
    if d.get(section) is None:
        d[section] = {}
    d[section].update(fields)


def put(key, value):
    if _current is not None:
        _current.data[key] = value


def add_input_file(path, role):
    if _current is None or path is None:
        return
    p = Path(path)
    if p.is_file():
        _current.data["inputs"]["files"].append(
            {"role": role, "path": str(p), "sha256": sha256_of(p)}
        )


def warn(message):
    if _current is not None and message not in _current.data["warnings"]:
        _current.data["warnings"].append(str(message))


def set_status(status, error=None):
    if _current is None:
        return
    _current.data["status"] = status
    if error and not _current.data["error"]:
        _current.data["error"] = str(error)


def finish(status=None, error=None):
    """Write the record. A run still 'running' becomes `status` (default ok); an error never
    overwrites an earlier failed/error status, and the first error message is kept."""
    global _current
    rec = _current
    if rec is None:
        return
    d = rec.data
    if d["status"] == "running":
        d["status"] = status or "ok"
    elif status == "error" and d["status"] in ("ok", "partial"):
        d["status"] = "error"
    if error and not d["error"]:
        d["error"] = str(error)
    try:
        rec.write()
    finally:
        _current = None
