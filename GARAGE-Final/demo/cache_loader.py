"""Shared Streamlit cache for demo_cache.pkl + expand_demo_cache (home + Diagnostics page)."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import streamlit as st

from demo_utils import expand_demo_cache, resolve_demo_cache_path


@st.cache_resource(ttl=3600, show_spinner=False)
def _cached_pickled(resolved: str, mtime_ns: int) -> Dict[str, Any]:
    with open(resolved, "rb") as f:
        return pickle.load(f)


def load_demo_expanded(demo_dir: Path) -> Tuple[Optional[Dict[str, Any]], Optional[Path]]:
    """Return (expanded context dict, path) or (None, None) if cache file is missing."""
    p = resolve_demo_cache_path(demo_dir)
    if not p.is_file():
        return None, None
    mtime_ns = int(p.stat().st_mtime_ns)
    raw = _cached_pickled(str(p.resolve()), mtime_ns)
    return expand_demo_cache(raw), p
