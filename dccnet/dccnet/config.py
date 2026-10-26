from __future__ import annotations

from pathlib import Path

import yaml


def _merge(base: dict, child: dict) -> dict:
    """Deep-merge child over base: dicts merge recursively, everything else is replaced."""
    out = dict(base)
    for k, v in child.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _load(path: Path, stack: tuple[Path, ...]) -> dict:
    path = Path(path).resolve()
    if path in stack:
        chain = " -> ".join(str(p) for p in (*stack, path))
        raise ValueError(f"Cyclic _base_ chain: {chain}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    base_ref = cfg.pop("_base_", None)
    if base_ref is None:
        return cfg
    base = _load(path.parent / base_ref, (*stack, path))
    return _merge(base, cfg)


def load_config(path: str | Path) -> dict:
    """Load a YAML config; `_base_: other.yaml` (relative to this file, chainable) is deep-merged under it."""
    return _load(Path(path), ())
