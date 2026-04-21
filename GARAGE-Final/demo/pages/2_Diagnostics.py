#!/usr/bin/env python3
"""Streamlit multipage: LLM diagnostics and ground-truth comparison for the selected window."""

import sys
import time
from pathlib import Path

_demo = Path(__file__).resolve().parent.parent
_root = _demo.parent
if str(_demo) not in sys.path:
    sys.path.insert(0, str(_demo))
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import requests
import streamlit as st

from cache_loader import load_demo_expanded
from demo_utils import (
    build_single_window_sensor_figure,
    check_fault_match,
    format_fault_type,
    gdn_flagged_sensor_names,
    get_lm_settings,
    normalize_lm_base_url,
    render_pyvis_graph,
)
from llm_diagnostic import run_live_diagnostic

st.set_page_config(
    page_title="GARAGE Diagnostics",
    layout="wide",
    page_icon="🚗",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .stApp { background-color: #0e1117; }
    .main { background-color: #0e1117; }
    .stPlotlyChart { background-color: #0e1117; }
    [data-testid="stSidebar"] { display: none; }
    [data-testid="collapsedControl"] { display: none; }
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: #1e293b; }
    ::-webkit-scrollbar-thumb { background: #475569; border-radius: 5px; }
    a:hover { text-decoration: none !important; }
</style>
""",
    unsafe_allow_html=True,
)

ctx, _demo_cache_path = load_demo_expanded(_demo)
if ctx is None:
    st.error(
        "Demo cache not found. Run:\n\n"
        "`python demo/build_demo_cache.py --checkpoint <path/to/stage2_clean_best.pt>`"
    )
    st.stop()

X_windows = ctx["X_windows"]
window_labels_true = ctx["window_labels_true"]
sensor_labels_true = ctx["sensor_labels_true"]
fault_types = ctx["fault_types"]
sensor_names = ctx["sensor_names"]
window_graphs = ctx["window_graphs"]
num_sensors = ctx["num_sensors"]
window_stats = ctx["window_stats"]
anomaly_scores = ctx["anomaly_scores"]
kg_contexts = ctx["kg_contexts"]
calibrated_thresholds = ctx["calibrated_thresholds"]
N = ctx["N"]
unique_drives = ctx["unique_drives"]
drive_to_indices = ctx["drive_to_indices"]
sensor_threshold = ctx["sensor_threshold"]
window_threshold = ctx["window_threshold"]
per_sensor_thresholds = ctx["per_sensor_thresholds"]
use_per_sensor_thresholds = ctx["use_per_sensor_thresholds"]
gt_window_fault = ctx["gt_window_fault"]
_window_score_fallback = ctx["window_score_fallback"]
gdn_window_flag = ctx["gdn_window_flag"]

if "selected_drive" not in st.session_state:
    st.session_state.selected_drive = unique_drives[0]
if "current_window" not in st.session_state:
    st.session_state.current_window = int(drive_to_indices[unique_drives[0]][0])
if "expert_mode" not in st.session_state:
    st.session_state.expert_mode = False
if "diagnostic_text" not in st.session_state:
    st.session_state.diagnostic_text = None
if "diagnostic_error" not in st.session_state:
    st.session_state.diagnostic_error = None
if "diagnostic_timestamp" not in st.session_state:
    st.session_state.diagnostic_timestamp = None
if "reveal_truth" not in st.session_state:
    st.session_state.reveal_truth = False
if "graph_html" not in st.session_state:
    st.session_state.graph_html = {}

_GRAPH_HTML_VERSION = 8
if st.session_state.get("_diag_graph_html_v") != _GRAPH_HTML_VERSION:
    st.session_state.graph_html = {}
    st.session_state._diag_graph_html_v = _GRAPH_HTML_VERSION

defaults = get_lm_settings()
if "lm_base_url" not in st.session_state:
    st.session_state.lm_base_url = defaults["base_url"]
if "lm_model" not in st.session_state:
    st.session_state.lm_model = defaults["model"]

qp = st.query_params
if "drive" in qp:
    dval = qp["drive"]
    if isinstance(dval, (list, tuple)):
        dval = dval[0]
    dval = str(dval)
    if dval in drive_to_indices:
        st.session_state.selected_drive = dval
if "w" in qp:
    try:
        wraw = qp["w"]
        if isinstance(wraw, (list, tuple)):
            wraw = wraw[0]
        w = int(wraw)
        if 0 <= w < N:
            st.session_state.current_window = w
    except (TypeError, ValueError):
        pass

idx_list = drive_to_indices.get(st.session_state.selected_drive, [0])
if st.session_state.current_window not in idx_list:
    st.session_state.current_window = int(idx_list[0])

st.session_state.selected_drive = str(st.session_state.selected_drive)
window_idx = int(st.session_state.current_window)


def ensure_window_graph_html(idx: int) -> str:
    gh = st.session_state.graph_html
    if idx not in gh:
        g = window_graphs.get(idx)
        if g is not None and g.number_of_nodes() > 0:
            sc_g = np.zeros(num_sensors)
            for ii in range(num_sensors):
                sname = sensor_names[ii]
                thr = (
                    per_sensor_thresholds.get(sname, sensor_threshold)
                    if use_per_sensor_thresholds
                    else sensor_threshold
                )
                sc_g[ii] = 1.0 if float(anomaly_scores[idx, ii]) > thr else 0.0
            gh[idx] = render_pyvis_graph(
                g, sensor_names, sc_g,
                window_status=int(gdn_window_flag[idx]),
                height="420px",
            )
        else:
            gh[idx] = '<div style="padding:2rem;color:#94a3b8;">No graph for this window</div>'
    return gh[idx]


st.title("Diagnostics")

_tb1, _tb2, _tb3, _tb4 = st.columns([1, 2, 2, 1])
with _tb1:
    st.session_state.expert_mode = st.toggle("Expert mode", value=st.session_state.expert_mode)
with _tb2:
    st.session_state.lm_base_url = st.text_input("LM URL", value=st.session_state.lm_base_url, label_visibility="collapsed", placeholder="LM Studio URL")
with _tb3:
    st.session_state.lm_model = st.text_input("Model", value=st.session_state.lm_model, label_visibility="collapsed", placeholder="Model id")
with _tb4:
    st.write("")
    if st.button("Test connection", use_container_width=True):
        try:
            url = normalize_lm_base_url(st.session_state.lm_base_url).rstrip("/") + "/chat/completions"
            r = requests.post(
                url,
                json={"model": st.session_state.lm_model, "messages": [{"role": "user", "content": "ping"}], "max_tokens": 2, "temperature": 0.0},
                timeout=15,
            )
            st.success("OK") if r.ok else st.error(f"{r.status_code}")
        except Exception as e:
            st.error(str(e)[:80])

st.divider()

st.subheader("Window detail")
gdn_alert = bool(gdn_window_flag[window_idx])
gdn_lbl, gdn_c = ("GDN anomaly", "#ef4444") if gdn_alert else ("GDN normal", "#22c55e")
st.markdown(
    f'<div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px;">'
    f'<div style="background-color:{gdn_c};padding:8px 16px;border-radius:6px;">'
    f'<span style="color:white;font-weight:bold;">{gdn_lbl} — Window {window_idx}</span></div>'
    f'<span style="color:#94a3b8;">Drive: {st.session_state.selected_drive}</span>'
    f"</div>",
    unsafe_allow_html=True,
)

graph_html = ensure_window_graph_html(window_idx)
if st.session_state.expert_mode:
    fig_sensors = build_single_window_sensor_figure(
        window_idx, X_windows, sensor_names, anomaly_scores,
        sensor_threshold, per_sensor_thresholds, use_per_sensor_thresholds,
    )
    fig_sensors.update_layout(title="Sensor traces (actual values)")
    lc, rc = st.columns([0.58, 0.42])
    with lc:
        st.plotly_chart(fig_sensors, use_container_width=True)
    with rc:
        st.markdown("##### Dependency graph")
        st.components.v1.html(graph_html, height=420, scrolling=False)
else:
    st.markdown("##### Sensor dependency graph")
    st.caption("Highlighted nodes are sensors whose readings exceeded the anomaly threshold for this window.")
    st.components.v1.html(graph_html, height=420, scrolling=False)

st.markdown("##### Sensor anomaly scores")
rows = []
for i, sname in enumerate(sensor_names):
    score = anomaly_scores[window_idx, i]
    thr = (
        per_sensor_thresholds.get(sname, sensor_threshold)
        if use_per_sensor_thresholds
        else sensor_threshold
    )
    rows.append({"Sensor": sname, "Score": f"{score:.3f}", "Threshold": f"{thr:.3f}", "Flag": "⚠️" if score > thr else "✓"})
st.dataframe(rows, use_container_width=True, hide_index=True)

st.markdown("---")
col_l, col_m, col_r = st.columns([0.34, 0.33, 0.33])
with col_l:
    st.markdown("#### Generate diagnostic")
    if st.button("Generate diagnostic", type="primary", use_container_width=True):
        st.session_state.diagnostic_error = None
        kg_ctx = kg_contexts.get(window_idx)
        ws_stats = window_stats.get(window_idx)
        if kg_ctx is None or ws_stats is None:
            st.session_state.diagnostic_error = "Missing KG context or window stats for this window."
        else:
            try:
                with st.spinner("Calling LM Studio…"):
                    diag = run_live_diagnostic(
                        window_idx=window_idx,
                        sensor_names=sensor_names,
                        kg_context=kg_ctx,
                        window_stats=ws_stats,
                        sensor_threshold=sensor_threshold,
                        sensor_thresholds=per_sensor_thresholds if use_per_sensor_thresholds else None,
                        expert=st.session_state.expert_mode,
                        base_url=st.session_state.lm_base_url,
                        model=st.session_state.lm_model,
                        anomaly_scores_row=anomaly_scores[window_idx],
                        gdn_window_alert=gdn_alert,
                    )
                    st.session_state.diagnostic_text = diag
                    st.session_state.diagnostic_timestamp = time.time()
                    st.session_state.reveal_truth = True
            except Exception as e:
                st.session_state.diagnostic_text = None
                st.session_state.reveal_truth = False
                st.session_state.diagnostic_error = str(e)
    if st.session_state.diagnostic_error:
        st.error(st.session_state.diagnostic_error)
        st.caption("Check LM Studio is running, CORS enabled, base URL ends with /v1, and model id matches the loaded model.")

with col_m:
    st.markdown("#### Model output")
    d = st.session_state.diagnostic_text
    if d:
        conf = str(d.get("confidence", "")).upper()
        colors = {"HIGH": "#22c55e", "MEDIUM": "#f59e0b", "LOW": "#ef4444"}
        cc = colors.get(conf, "#64748b")
        fs = ", ".join(d.get("faulty_sensors") or []) or "None"
        fault_type_val = d.get("fault_type", "")
        reasoning_val = d.get("reasoning", "")

        if not st.session_state.expert_mode:
            is_normal = fault_type_val.lower() == "normal" if fault_type_val else True
            if is_normal:
                summary_html = (
                    '<p style="color:#4ade80;font-size:15px;font-weight:600;margin-bottom:8px;">'
                    "The vehicle appears to be operating normally for this window."
                    "</p>"
                )
            else:
                faulty_display = fs if fs != "None" else "one or more sensors"
                summary_html = (
                    f'<p style="color:#f87171;font-size:15px;font-weight:600;margin-bottom:8px;">'
                    f"A fault was detected involving {faulty_display}."
                    f"</p>"
                )
        else:
            summary_html = ""

        st.markdown(
            f"""
<div style="background-color:#1e293b;padding:16px;border-radius:8px;border:1px solid #334155;">
{summary_html}
<p style="color:#94a3b8;font-size:13px;margin-bottom:2px;">{"Affected sensors" if not st.session_state.expert_mode else "Faulty sensors"}</p>
<p style="color:#e2e8f0;">{fs}</p>
<p style="color:#94a3b8;font-size:13px;margin-bottom:2px;">{"Fault category" if not st.session_state.expert_mode else "Fault type"}</p>
<p style="color:#e2e8f0;font-weight:500;">{fault_type_val}</p>
<p style="color:#94a3b8;font-size:13px;margin-bottom:2px;">Confidence</p>
<p><span style="background-color:{cc};color:white;padding:2px 8px;border-radius:4px;font-weight:bold;">{conf}</span></p>
<p style="color:#94a3b8;font-size:13px;margin-bottom:2px;">{"Explanation" if not st.session_state.expert_mode else "Reasoning"}</p>
<p style="color:#e2e8f0;white-space:pre-wrap;">{reasoning_val}</p>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.session_state.diagnostic_timestamp:
            st.caption(f"Generated {time.time() - st.session_state.diagnostic_timestamp:.1f}s ago")
    else:
        st.info("Run **Generate diagnostic** to call the LM.")

with col_r:
    st.markdown("#### Ground truth (reference)")
    if st.session_state.reveal_truth and st.session_state.diagnostic_text:
        wl_gt_fault = bool(gt_window_fault[window_idx])
        st.markdown(
            f'<p style="color:#94a3b8;font-size:14px;margin-bottom:4px;">Window-level (GT)</p>'
            f'<p style="color:#e2e8f0;font-weight:500;margin-top:0;">'
            f'{"Faulty" if wl_gt_fault else "Normal"}</p>',
            unsafe_allow_html=True,
        )
        ft = fault_types[window_idx]
        faulty = [
            sensor_names[i]
            for i, lab in enumerate(sensor_labels_true[window_idx])
            if lab == 1
        ]
        pred_sensors = list(st.session_state.diagnostic_text.get("faulty_sensors") or [])
        gdn_list = gdn_flagged_sensor_names(
            anomaly_scores[window_idx], list(sensor_names),
            sensor_threshold, per_sensor_thresholds if use_per_sensor_thresholds else None,
        )
        pred_set, gt_set, gdn_set = set(pred_sensors), set(faulty), set(gdn_list)
        if pred_set == gt_set:
            sens_note = "Faulty sensors: **exact match** vs ground truth."
        elif pred_set & gt_set:
            sens_note = f"Faulty sensors: **partial overlap** (pred {sorted(pred_set)}, GT {sorted(gt_set)})."
        else:
            sens_note = f"Faulty sensors: **no overlap** with GT (pred {sorted(pred_set)}, GT {sorted(gt_set)})."
        if gdn_set:
            sens_note += f" GDN-flagged: {sorted(gdn_set)}."
        st.caption(sens_note)
        pred_ft = st.session_state.diagnostic_text.get("fault_type", "")
        match_status = check_fault_match(ft, pred_ft)
        if match_status == "correct":
            badge_c, badge_t = "#22c55e", "Match"
        elif match_status == "partial":
            badge_c, badge_t = "#f59e0b", "Partial"
        else:
            badge_c, badge_t = "#ef4444", "No match"
        st.markdown(
            f"""
<div style="background-color:#1e293b;padding:16px;border-radius:8px;border:1px solid #334155;">
<div style="background-color:{badge_c};display:inline-block;padding:4px 12px;border-radius:4px;margin-bottom:8px;">
<span style="color:white;font-weight:bold;">{badge_t}</span></div>
<p style="color:#94a3b8;font-size:14px;">Fault type</p>
<p style="color:#e2e8f0;">{format_fault_type(ft)}</p>
<p style="color:#94a3b8;font-size:14px;">Faulty sensors</p>
<p style="color:#e2e8f0;">{", ".join(faulty) or "None"}</p>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.caption("Generate a diagnostic to compare with ground truth.")
