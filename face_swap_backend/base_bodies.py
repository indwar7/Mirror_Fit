"""
The curated base-body asset set.

Base bodies are photographs, committed to the repo, each one reviewed by a
person before it was admitted. Nothing in this module — or anywhere else in the
runtime — creates one.

Why not generate them
---------------------
They used to be generated at build time by a text-to-image model. Generation is
stochastic: a fraction of renders came out with two torsos, and the gate in
front of them asked only "is a face detectable?", which a two-torso image
answers yes. Eighteen broken figures shipped, twice. Better models lower the
rate; they do not turn the output into something you can ship without a human
looking at every frame. So a human looks at every frame, once, offline, and the
result is an asset.

No fallbacks
------------
If a bin has no photograph, `get()` raises MissingBaseBody and the endpoint
returns 501 naming the id. It does not fall back to a neighbouring build, and
it does not synthesise anything. A silent substitution is how someone ends up
being shown a body that is not the one their measurements asked for, with
nothing in the response saying so.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Optional

import body_shapes

ASSET_DIR = pathlib.Path(__file__).parent.parent / "assets" / "base_bodies"
MANIFEST_PATH = ASSET_DIR / "manifest.json"

MANIFEST_VERSION = 1


class MissingBaseBody(LookupError):
    """No approved photograph for this bin. Callers must surface it, not paper over it."""


class CorruptBaseBody(RuntimeError):
    """The file on disk does not match the sha256 the manifest recorded."""


@dataclass(frozen=True)
class BaseBody:
    id: str
    gender: str
    build: str
    height_cm: float
    license: str
    source: str
    sha256: str
    file: str
    approved_by: str
    approved_at: str

    def path(self) -> pathlib.Path:
        return ASSET_DIR / self.file


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"version": MANIFEST_VERSION, "bodies": []}
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        # Unlike the user-avatar store, a broken manifest is NOT degraded to
        # "empty". Empty here would mean every avatar silently 501s and the
        # cause would look like missing assets rather than a corrupt file.
        raise RuntimeError(f"base_bodies manifest unreadable: {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("bodies"), list):
        raise RuntimeError("base_bodies manifest has the wrong shape")
    return data


def all_bodies() -> dict[str, BaseBody]:
    out: dict[str, BaseBody] = {}
    for row in _load_manifest()["bodies"]:
        try:
            body = BaseBody(**row)
        except TypeError as e:
            raise RuntimeError(f"base_bodies manifest entry is malformed: {e}") from e
        out[body.id] = body
    return out


def sha256_of(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get(body_id: str, verify: bool = True) -> BaseBody:
    """The approved photograph for `body_id`.

    Raises MissingBaseBody if it is not in the manifest or not on disk, and
    CorruptBaseBody if the bytes no longer match what was approved — an image
    swapped after review has not been reviewed.
    """
    body = all_bodies().get(body_id)
    if body is None:
        raise MissingBaseBody(
            f"No approved base body for '{body_id}'. Add one with "
            f"tools/curate_base_bodies.py; nothing is generated at runtime."
        )
    path = body.path()
    if not path.exists():
        raise MissingBaseBody(
            f"Base body '{body_id}' is in the manifest but '{body.file}' is "
            f"not on disk."
        )
    if verify and sha256_of(path) != body.sha256:
        raise CorruptBaseBody(
            f"Base body '{body_id}' does not match its approved sha256 — the "
            f"file changed after review. Re-run curation on it."
        )
    return body


def status() -> dict:
    """Per-id presence, for the operator and for the acceptance check.

    Reports each id separately on purpose: 'all bodies present' is the shape of
    claim that let two rounds of broken assets through.
    """
    expected = body_shapes.all_base_body_ids()
    manifest = all_bodies()
    rows = []
    for body_id in expected:
        body = manifest.get(body_id)
        if body is None:
            rows.append({"id": body_id, "state": "missing", "detail": "not in manifest"})
            continue
        path = body.path()
        if not path.exists():
            rows.append({"id": body_id, "state": "missing", "detail": f"{body.file} absent"})
            continue
        if sha256_of(path) != body.sha256:
            rows.append({"id": body_id, "state": "corrupt", "detail": "sha256 mismatch"})
            continue
        rows.append({
            "id": body_id, "state": "ok",
            "license": body.license, "approved_by": body.approved_by,
        })
    return {
        "expected": expected,
        "bodies": rows,
        "ready": sum(1 for r in rows if r["state"] == "ok"),
        "total": len(expected),
    }
