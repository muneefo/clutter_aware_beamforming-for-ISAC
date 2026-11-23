"""Archive all Part E labels and teacher traces in small, verifiable chunks."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from parte_teacher import TRAIN


def bundle(teacher, output):
    roots = sorted(p for p in teacher.glob("parte-train-*") if p.is_dir())
    inputs = {}
    for root in roots:
        rows = [json.loads(line) for line in (root / "teacher.jsonl").read_text().splitlines()]
        for row in rows:
            sid = row["scene_id"]
            if sid in inputs:
                raise ValueError(f"Duplicate teacher scene {sid}")
            path = root / f"{sid}.npz"
            if row["status"] != "converged" or not path.is_file():
                raise ValueError(f"Scene {sid} lacks a converged teacher label")
            inputs[sid] = path
    if inputs.keys() != set(TRAIN):
        raise ValueError("Teacher coverage differs from the registered 5,000 scenes")

    output.mkdir(parents=True, exist_ok=False)
    labels = output / "labels"
    labels.mkdir()
    entries = []
    for lane, root in enumerate(roots):
        target = output / f"teacher-{lane}"
        target.mkdir()
        for name in ("teacher.jsonl", "environment.json"):
            shutil.copyfile(root / name, target / name)
        paths = sorted((p for p in inputs.values() if p.parent == root), key=lambda p: p.name)
        for start in range(0, len(paths), 250):
            chunk = labels / f"teacher-{lane}-{start//250:02}.zip"
            with ZipFile(chunk, "x", compression=ZIP_DEFLATED, compresslevel=1) as zipped:
                for path in paths[start:start+250]:
                    data = path.read_bytes()
                    member = f"{root.name}/{path.name}"
                    zipped.writestr(member, data)
                    entries.append(dict(scene_id=int(path.stem), archive=chunk.name,
                                        member=member, sha256=hashlib.sha256(data).hexdigest()))
            with ZipFile(chunk) as zipped:
                if zipped.testzip() is not None:
                    raise IOError(f"Corrupt archive {chunk}")
    (labels / "manifest.jsonl").write_text("".join(
        json.dumps(r) + "\n" for r in sorted(entries, key=lambda r: r["scene_id"])))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("teacher", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    bundle(args.teacher, args.output)
