"""
Live LM Studio diagnostic calls (demo-only). Uses build_kag_prompt from eval + injection fault block.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_demo_dir = Path(__file__).resolve().parent
if str(_demo_dir) not in sys.path:
    sys.path.insert(0, str(_demo_dir))
_parent = _demo_dir.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from injection_catalog import format_injection_block_for_prompt
from prompts import SYSTEM_PROMPT_NON_EXPERT
from schemas import DemoFaultDiagnosis

# Import eval prompt builder (read-only dependency on repo)
from llm.evaluation.evaluate_gdn_kg_llm import SYSTEM_PROMPT, build_kag_prompt


def _per_sensor_thr(
    name: str,
    sensor_threshold: float,
    sensor_thresholds: Optional[Dict[str, float]],
) -> float:
    if sensor_thresholds:
        return float(sensor_thresholds.get(name, sensor_threshold))
    return float(sensor_threshold)


def _gdn_flagged_sensor_names(
    scores_1d: np.ndarray,
    sensor_names: List[str],
    sensor_threshold: float,
    sensor_thresholds: Optional[Dict[str, float]] = None,
) -> List[str]:
    flagged: List[Tuple[str, float]] = []
    for i, name in enumerate(sensor_names):
        s = float(scores_1d[i])
        if s > _per_sensor_thr(name, sensor_threshold, sensor_thresholds):
            flagged.append((name, s))
    flagged.sort(key=lambda x: -x[1])
    return [n for n, _ in flagged]


def _normalize_lm_base_url(url: str) -> str:
    """Ensure OpenAI-compatible root ends with /v1 (matches demo_utils.normalize_lm_base_url)."""
    u = str(url).strip().rstrip("/")
    if not u.endswith("/v1"):
        u = u + "/v1"
    return u


def kg_context_to_violations_and_propagation(
    kg_context: Dict[str, Any],
) -> Tuple[List[Tuple[str, str, float, float]], List[str]]:
    violations: List[Tuple[str, str, float, float]] = []
    for rel in kg_context.get("violations", []):
        a = rel.get("source", "")
        b = rel.get("target", "")
        exp = float(
            rel.get(
                "expected_correlation_gdn",
                rel.get("expected_correlation", 0),
            )
        )
        act = float(rel.get("correlation", 0))
        if a and b:
            violations.append((a, b, exp, act))

    propagation_chain: List[str] = []
    for prop in kg_context.get("anomaly_propagation", []):
        root = prop.get("root_sensor", "")
        affected = prop.get("affected_sensors", [])
        if root:
            propagation_chain = [root] + list(affected)
            break

    return violations, propagation_chain


def window_stats_to_sensor_scores(window_stats: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, stats in window_stats.items():
        if hasattr(stats, "anomaly_score"):
            out[name] = float(stats.anomaly_score)
        elif isinstance(stats, dict):
            out[name] = float(stats.get("anomaly_score", 0.0))
    return out


def format_raw_gdn_block(
    sensor_names: List[str],
    scores_1d: np.ndarray,
    sensor_threshold: float,
    sensor_thresholds: Optional[Dict[str, float]],
    gdn_window_alert: bool = True,
) -> str:
    """Explicit per-sensor GDN scores so the LM aligns faulty_sensors with detector output."""
    lines = [
        "RAW GDN ANOMALY SCORES (sigmoid 0–1; same thresholds as ANOMALOUS SENSORS list):",
    ]
    for i, name in enumerate(sensor_names):
        s = float(scores_1d[i])
        thr = _per_sensor_thr(name, sensor_threshold, sensor_thresholds)
        flag = " **ANOMALOUS**" if s > thr else ""
        lines.append(f"  {name}: score={s:.3f}, threshold={thr:.3f}{flag}")
    if gdn_window_alert:
        lines.append(
            "You must include every **ANOMALOUS** sensor in faulty_sensors unless reasoning states a clear false positive."
        )
    else:
        lines.append(
            "This window is **below GDN window alert**: isolated **ANOMALOUS** tags are often borderline noise. "
            "Prefer **fault_type \"normal\"** and **faulty_sensors []** unless several sensors clearly exceed threshold; "
            "justify any fault in reasoning."
        )
    return "\n".join(lines)


def strip_eval_fault_line(user_content: str) -> str:
    """Remove the eval script's single-line FAULT_TYPE hint; injection block is appended separately."""
    lines = user_content.split("\n")
    out: List[str] = []
    for line in lines:
        if line.strip().startswith("FAULT_TYPE:"):
            continue
        out.append(line)
    return "\n".join(out).rstrip()


def build_demo_messages(
    window_idx: int,
    sensor_names: List[str],
    kg_context: Dict[str, Any],
    window_stats: Dict[str, Any],
    sensor_threshold: float,
    sensor_thresholds: Optional[Dict[str, float]],
    expert: bool,
    gdn_scores_row: np.ndarray,
    gdn_window_alert: bool = True,
) -> List[Dict[str, str]]:
    sensor_scores = window_stats_to_sensor_scores(window_stats)
    violations, propagation_chain = kg_context_to_violations_and_propagation(kg_context)

    base = build_kag_prompt(
        window_idx,
        sensor_scores,
        violations,
        propagation_chain,
        sensor_names,
        sensor_threshold,
        sensor_thresholds=sensor_thresholds,
    )
    user_body = strip_eval_fault_line(base[1]["content"])
    user_body = user_body + "\n\n" + format_raw_gdn_block(
        sensor_names,
        gdn_scores_row,
        sensor_threshold,
        sensor_thresholds,
        gdn_window_alert=gdn_window_alert,
    )
    user_body = user_body + "\n\n" + format_injection_block_for_prompt()

    if not gdn_window_alert:
        user_body += (
            "\n\nWINDOW-LEVEL CONTEXT: GDN **global window score** is **below** the window alert threshold "
            "(this window is treated as **no window-level alert**). Per-sensor lines can still kiss the threshold "
            "from noise. Prefer **fault_type \"normal\"** and **faulty_sensors []** unless several sensors clearly "
            "and persistently exceed threshold with a coherent pattern; if you choose normal, say so in reasoning."
        )

    system = SYSTEM_PROMPT if expert else SYSTEM_PROMPT_NON_EXPERT
    if gdn_window_alert:
        system += (
            "\n\nDemo alignment: the user message includes RAW GDN ANOMALY SCORES. "
            "faulty_sensors must list every sensor marked **ANOMALOUS** there unless reasoning argues a false positive. "
            "The JSON field `reasoning` is required: at least two sentences naming sensors and numeric scores from that block; "
            "do not leave `reasoning` empty."
        )
    else:
        system += (
            "\n\nDemo alignment: This window is **below GDN window alert** (normal at window level). "
            "Do **not** force a fault label from borderline single-sensor scores. "
            "Default to **fault_type \"normal\"** and **faulty_sensors []** unless evidence is strong. "
            "The JSON field `reasoning` is required: at least two sentences referencing RAW GDN SCORES "
            "(e.g. why scores are noise/borderline vs a real fault); do not leave `reasoning` empty."
        )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_body},
    ]


def call_lm_studio_json_schema(
    messages: List[Dict[str, str]],
    base_url: str,
    model: str,
    timeout: int = 300,
) -> str:
    base_url = _normalize_lm_base_url(base_url)
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 4096,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "DemoFaultDiagnosis",
                "strict": True,
                "schema": DemoFaultDiagnosis.model_json_schema(),
            },
        },
    }
    resp = requests.post(url, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def merge_faulty_sensors_with_gdn(
    parsed: Dict[str, Any],
    sensor_names: List[str],
    gdn_scores_row: np.ndarray,
    sensor_threshold: float,
    sensor_thresholds: Optional[Dict[str, float]],
    *,
    gdn_window_alert: bool = True,
) -> None:
    """
    If the LM omits GDN-flagged sensors, union them in (demo alignment with detector).

    When ``gdn_window_alert`` is False, merging is skipped: per-sensor thresholds often
    fire on noise for GT-normal windows, and union would force bogus faults vs labels.
    """
    if not gdn_window_alert:
        return
    flagged = _gdn_flagged_sensor_names(
        gdn_scores_row, sensor_names, sensor_threshold, sensor_thresholds
    )
    if not flagged:
        return
    cur = list(parsed.get("faulty_sensors") or [])
    if not cur:
        parsed["faulty_sensors"] = flagged
        return
    idx = {n: i for i, n in enumerate(sensor_names)}
    missing = [n for n in flagged if n not in cur]
    if missing:
        merged = sorted(set(cur) | set(flagged), key=lambda n: -float(gdn_scores_row[idx[n]]))
        parsed["faulty_sensors"] = merged


def parse_demo_diagnostic(raw: str) -> Dict[str, Any]:
    text = raw.strip()
    try:
        data = DemoFaultDiagnosis.model_validate_json(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}\s*$", text)
        if not m:
            raise
        data = DemoFaultDiagnosis.model_validate_json(m.group(0))
    return {
        "faulty_sensors": list(data.faulty_sensors),
        "fault_type": data.fault_type,
        "confidence": data.confidence,
        "reasoning": data.reasoning,
    }


def run_live_diagnostic(
    window_idx: int,
    sensor_names: List[str],
    kg_context: Dict[str, Any],
    window_stats: Dict[str, Any],
    sensor_threshold: float,
    sensor_thresholds: Optional[Dict[str, float]],
    expert: bool,
    base_url: str,
    model: str,
    anomaly_scores_row: np.ndarray,
    gdn_window_alert: bool = True,
) -> Dict[str, Any]:
    gdn_row = np.asarray(anomaly_scores_row, dtype=np.float64).reshape(-1)
    messages = build_demo_messages(
        window_idx,
        sensor_names,
        kg_context,
        window_stats,
        sensor_threshold,
        sensor_thresholds,
        expert=expert,
        gdn_scores_row=gdn_row,
        gdn_window_alert=gdn_window_alert,
    )
    raw = call_lm_studio_json_schema(messages, base_url, model)
    parsed = parse_demo_diagnostic(raw)
    merge_faulty_sensors_with_gdn(
        parsed,
        sensor_names,
        gdn_row,
        sensor_threshold,
        sensor_thresholds,
        gdn_window_alert=gdn_window_alert,
    )
    return parsed
