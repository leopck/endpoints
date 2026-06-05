#!/usr/bin/env python3
"""Rebuild preds.json from per-instance preds_entry.json files.

Usage: rebuild_preds.py <output_dir> [<model_name>]
"""
import json
import sys
from pathlib import Path

d = Path(sys.argv[1])
model = sys.argv[2] if len(sys.argv) > 2 else "model"
preds = []
for inst in sorted(d.iterdir()):
    if not inst.is_dir():
        continue
    pe = inst / "preds_entry.json"
    if pe.exists():
        preds.append(json.loads(pe.read_text()))
    else:
        preds.append({
            "instance_id": inst.name,
            "model_name_or_path": model,
            "model_patch": "",
        })
out_path = d / "preds.json"
out_path.write_text(json.dumps(preds, indent=2))
n_with = sum(1 for p in preds if (p.get("model_patch") or "").strip())
print(f"{d.name}: total={len(preds)} with_patch={n_with} -> {out_path}")
