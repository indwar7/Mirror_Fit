"""
User-enrolled avatars — the shopper's own face becomes an avatar.

Why a separate store
--------------------
The preset catalogue in main.py is a hard-coded list, generated offline by
generate_avatars.py and fixed at import time. Enrolment needs the opposite:
records created at runtime, one per shopper. Keeping them in their own JSON
sidecar means the presets stay immutable and a corrupt/absent store degrades
to "presets only" rather than taking the catalogue down.

Why this is all it takes
------------------------
Every downstream consumer — static swap, live swap V1/V2, Wav2Lip lipsync,
hair transfer — resolves an avatar the same way: `avatars_cache/{id}.jpg`.
Writing an enrolled face to that path under a `usr_` id makes it work
everywhere without touching those code paths.

Reserved fields
---------------
`body_image` and `measurements` are written as null. They are for the
full-body avatar (an SD-generated figure with the enrolled face swapped onto
it) and the chest/waist/shoulder fit pass. Declaring them now means adding
those stages later is a write, not a migration.

Voice note: `_transform_voice` in main.py picks its pitch shift by id prefix
and falls through to 0 semitones for anything unrecognised. A `usr_` avatar
therefore keeps the speaker's natural voice, which is the correct default
when the avatar *is* the speaker. No prefix rule is wanted here.
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import tempfile
import threading
import time
import uuid
from typing import Optional

# Schema version, so a later field change can be detected rather than guessed.
STORE_VERSION = 1

ID_PREFIX = "usr_"
CATEGORY = "You"

_lock = threading.Lock()


def _store_path(cache_dir: pathlib.Path) -> pathlib.Path:
    return cache_dir / "user_avatars.json"


def new_id() -> str:
    """Short, collision-safe, and visibly distinct from preset `gen_`/`ai_` ids."""
    return ID_PREFIX + uuid.uuid4().hex[:8]


def load(cache_dir: pathlib.Path) -> list[dict]:
    """Every enrolled record, oldest first. Never raises — a broken store
    reads as empty so the preset catalogue still serves."""
    path = _store_path(cache_dir)
    if not path.exists():
        return []
    try:
        with _lock:
            data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    records = data.get("avatars")
    return records if isinstance(records, list) else []


def _write(cache_dir: pathlib.Path, records: list[dict]) -> None:
    """Atomic replace — a crash mid-write must not truncate the catalogue."""
    path = _store_path(cache_dir)
    payload = json.dumps(
        {"version": STORE_VERSION, "avatars": records}, indent=2
    )
    fd, tmp = tempfile.mkstemp(dir=str(cache_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def make_record(
    avatar_id: str,
    name: str,
    gender: Optional[str],
    face_image: str,
) -> dict:
    return {
        "id": avatar_id,
        "name": name,
        "category": CATEGORY,
        "gender": gender,          # "male" / "female" / None when undetected
        "created_at": int(time.time()),
        "face_image": face_image,  # filename inside avatars_cache/
        # ── Reserved: see module docstring ──
        "body_image": None,        # SD full-body figure + enrolled face
        "measurements": None,      # {"chest_cm", "waist_cm", "shoulder_cm", ...}
    }


def add(cache_dir: pathlib.Path, record: dict) -> dict:
    """Append a record. Replaces any existing record with the same id."""
    with _lock:
        path = _store_path(cache_dir)
        records = []
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("avatars"), list):
                    records = data["avatars"]
            except (OSError, ValueError):
                records = []
        records = [r for r in records if r.get("id") != record["id"]]
        records.append(record)
        _write(cache_dir, records)
    return record


def update(cache_dir: pathlib.Path, avatar_id: str, fields: dict) -> Optional[dict]:
    """Merge `fields` into one record. Returns the updated record, or None if
    the id was not enrolled.

    Read-modify-write under the same lock as add/remove, so a body generation
    landing at the same time as another write cannot lose either change.
    """
    with _lock:
        path = _store_path(cache_dir)
        records = []
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("avatars"), list):
                    records = data["avatars"]
            except (OSError, ValueError):
                records = []

        target = next((r for r in records if r.get("id") == avatar_id), None)
        if target is None:
            return None

        target.update(fields)
        _write(cache_dir, records)
        return dict(target)


def get(cache_dir: pathlib.Path, avatar_id: str) -> Optional[dict]:
    for r in load(cache_dir):
        if r.get("id") == avatar_id:
            return r
    return None


def remove(cache_dir: pathlib.Path, avatar_id: str) -> bool:
    """Drop a record and its image files. False if the id was not enrolled."""
    with _lock:
        records = []
        path = _store_path(cache_dir)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("avatars"), list):
                    records = data["avatars"]
            except (OSError, ValueError):
                records = []
        target = next((r for r in records if r.get("id") == avatar_id), None)
        if target is None:
            return False
        _write(cache_dir, [r for r in records if r.get("id") != avatar_id])

    for key in ("face_image", "body_image"):
        filename = target.get(key)
        if filename:
            with contextlib.suppress(OSError):
                (cache_dir / filename).unlink()
    return True
