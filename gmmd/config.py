"""Configs for GMMD experiments: ``configs/<name>.toml``, with ``inherit = "<base>"`` variants.

A config file is plain TOML.  A variant declares ``inherit = "<base>"`` and deep-merges its own
keys onto that base, so a variant file states only what it changes; chains work, cycles raise.
Names may contain ``/`` and the results directory mirrors them under ``results/``.
"""

from __future__ import annotations

import json
from pathlib import Path


__all__ = ["CONFIG_DIR", "RESULTS_DIR", "config_path", "load_config", "resolve_path",
           "parse_overrides"]

_REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = _REPO / "configs"
RESULTS_DIR = _REPO / "results"


def _load_toml(path):
    import tomllib

    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _deep_copy(d):
    return {k: _deep_copy(v) if isinstance(v, dict) else v for k, v in d.items()}


def _deep_merge(base, extra):
    """``extra`` merged onto ``base``, recursing into dicts (a scalar replaces a scalar)."""
    out = _deep_copy(base)
    for k, v in extra.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def config_path(name):
    for stem in (name, f"{name}/base"):
        p = CONFIG_DIR / f"{stem}.toml"
        if p.exists():
            return p
    return None


def load_config(name, overrides=None, _seen=()):
    """Resolved settings of ``configs/<name>.toml`` (variants merged onto their base)."""
    p = config_path(name)
    if p is None:
        raise FileNotFoundError(f"no GMMD config {name!r} under {CONFIG_DIR}")
    cfg = _deep_copy(_load_toml(p))
    base = cfg.pop("inherit", None)
    if base:
        if name in _seen:
            raise ValueError("config inheritance cycle: " + " -> ".join((*_seen, name, base)))
        cfg = _deep_merge(load_config(base, _seen=(*_seen, name)), cfg)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    cfg["name"] = name
    return cfg


def resolve_path(value, base):
    """A config path: absolute as is, else relative to ``base``."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else Path(base) / p


def parse_overrides(pairs):
    """``["a.b=1", "c=x"]`` -> nested dict, values parsed as JSON when possible."""
    out = {}
    for pair in pairs or ():
        key, _, raw = pair.partition("=")
        if not _:
            raise ValueError(f"override {pair!r} is not key=value")
        try:
            val = json.loads(raw)
        except json.JSONDecodeError:
            val = raw
        d = out
        *parents, leaf = key.split(".")
        for k in parents:
            d = d.setdefault(k, {})
        d[leaf] = val
    return out
