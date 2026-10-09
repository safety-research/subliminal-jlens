"""Small I/O helpers shared by the scripts."""
import hashlib
import json
import os


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def stable_int(*parts, mod=2 ** 31 - 1):
    """Per-request sampling seed used for every vLLM generation in this project: seed = f(global seed, request id).
    Identical to the rule of the (private) generation tool the original runs used, so request seeds are reproducible."""
    h = hashlib.sha1("\x1f".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:12], 16) % mod


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data")
# Where generated data, checkpoints, readouts and results go. Override with SLJ_OUT.
OUT = os.environ.get("SLJ_OUT", os.path.join(REPO, "outputs"))
# Where lens files live. Override with SLJ_LENS_DIR.
LENS_DIR = os.environ.get("SLJ_LENS_DIR", os.path.join(OUT, "lenses"))


def out(*parts):
    p = os.path.join(OUT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
