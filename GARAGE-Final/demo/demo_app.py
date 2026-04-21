#!/usr/bin/env python3
"""
GARAGE-Final Streamlit Demo — drive-first UI, live LM Studio diagnostics (demo/ only).
"""

import sys
from pathlib import Path

_demo = Path(__file__).resolve().parent
_root = _demo.parent
if str(_demo) not in sys.path:
    sys.path.insert(0, str(_demo))
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import html
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from cache_loader import load_demo_expanded
from demo_utils import (
    apply_dark_theme,
    get_lm_settings,
)

st.set_page_config(
    page_title="GARAGE Demo",
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
    div[data-testid="stHorizontalBlock"] button p {
        white-space: nowrap !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind="primary"],
    div[data-testid="stHorizontalBlock"] button[data-testid="baseButton-primary"] {
        background-color: #b91c1c !important;
        border-color: #991b1b !important;
        color: #ffffff !important;
    }
    div[data-testid="stHorizontalBlock"] button[kind="secondary"],
    div[data-testid="stHorizontalBlock"] button[data-testid="baseButton-secondary"] {
        background-color: #15803d !important;
        border-color: #166534 !important;
        color: #ffffff !important;
    }
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
window_stats = ctx["window_stats"]
anomaly_scores = ctx["anomaly_scores"]
kg_contexts = ctx["kg_contexts"]
calibrated_thresholds = ctx["calibrated_thresholds"]
N = ctx["N"]
num_sensors = ctx["num_sensors"]
drive_ids = ctx["drive_ids"]
unique_drives = ctx["unique_drives"]
drive_to_indices = ctx["drive_to_indices"]
synthetic_drives = ctx["synthetic_drives"]
sensor_threshold = ctx["sensor_threshold"]
window_threshold = ctx["window_threshold"]
per_sensor_thresholds = ctx["per_sensor_thresholds"]
use_per_sensor_thresholds = ctx["use_per_sensor_thresholds"]
gt_window_fault = ctx["gt_window_fault"]
window_scores = ctx["window_scores"]
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
window_idx = st.session_state.current_window

st.title("GARAGE: GrAph-based Reasoning for Automotive diagnostics GEneration")

_tb1, _tb2, _tb3, _tb4 = st.columns([1, 2, 2, 1])
with _tb1:
    st.session_state.expert_mode = st.toggle("Expert mode", value=st.session_state.expert_mode)
with _tb2:
    st.session_state.lm_base_url = st.text_input("LM Studio URL", value=st.session_state.lm_base_url, label_visibility="collapsed", placeholder="LM Studio URL")
with _tb3:
    st.session_state.lm_model = st.text_input("Model id", value=st.session_state.lm_model, label_visibility="collapsed", placeholder="Model id")
with _tb4:
    st.write("")
    if st.button("Test connection", use_container_width=True):
        import requests
        from demo_utils import normalize_lm_base_url
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

palette = [
    "#38bdf8", "#a78bfa", "#f472b6", "#fbbf24",
    "#4ade80", "#fb923c", "#2dd4bf", "#e879f9",
]


def _short_sensor_label(name: str) -> str:
    s = str(name).replace(" ()", "").strip()
    return s[:24] + "…" if len(s) > 24 else s


def build_drive_traces_figure(
    drive_indices: list,
    max_windows: int,
    chart_title: str = "",
) -> go.Figure:
    """Concatenate windows along time; light GDN tint per window so traces stay readable."""
    use_idx = drive_indices[:max_windows]
    if not use_idx:
        return go.Figure()
    T = int(X_windows.shape[1])
    n_s = len(sensor_names)
    total_len = int(len(use_idx) * T)
    max_pts = 16_000
    step = max(1, (total_len + max_pts - 1) // max_pts)
    x_idx = np.arange(0, total_len, step, dtype=np.float64)
    fig = make_subplots(rows=n_s, cols=1, shared_xaxes=True, vertical_spacing=0.045)
    for si in range(n_s):
        y = np.concatenate([X_windows[gw, :, si] for gw in use_idx])
        y_ds = y[::step]
        m = min(int(x_idx.shape[0]), int(y_ds.shape[0]))
        c = palette[si % len(palette)]
        fig.add_trace(
            go.Scattergl(
                x=x_idx[:m], y=y_ds[:m], mode="lines",
                name=sensor_names[si], showlegend=False,
                line=dict(color=c, width=2),
                hovertemplate=f"{sensor_names[si]}<br>t=%{{x}} value=%{{y:.3f}}<extra></extra>",
            ),
            row=si + 1, col=1,
        )
    band_shapes: list = []
    for k, gw in enumerate(use_idx):
        x0, x1 = k * T - 0.5, (k + 1) * T - 0.5
        if gdn_window_flag[gw]:
            fill, line = "rgba(239,68,68,0.12)", dict(color="rgba(248,113,113,0.55)", width=1)
        else:
            fill, line = "rgba(34,197,94,0.10)", dict(color="rgba(74,222,128,0.5)", width=1)
        band_shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=x0, x1=x1, y0=0, y1=1,
            fillcolor=fill, line=line, layer="below",
        ))
    theme = apply_dark_theme()["layout"]
    base = {k: theme[k] for k in ("paper_bgcolor", "plot_bgcolor", "font") if k in theme}
    row_h = max(96, min(118, 5200 // max(n_s, 1)))
    fig.update_layout(
        **base, shapes=band_shapes,
        height=max(380, row_h * n_s),
        margin=dict(l=8, r=28, t=56, b=40),
        title=dict(text=chart_title or "Sensor traces (each band = one window)", font=dict(size=14, color="#e2e8f0"), x=0, xanchor="left"),
        legend=dict(bgcolor="#1e293b", bordercolor="#334155", font=dict(color="#e2e8f0")),
    )
    for si in range(n_s):
        fig.update_yaxes(
            title_text=_short_sensor_label(sensor_names[si]),
            title_font=dict(size=11, color="#94a3b8"),
            automargin=True, showgrid=True,
            gridcolor="rgba(148,163,184,0.12)", zeroline=False,
            row=si + 1, col=1,
        )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.15)", zeroline=False)
    return fig


WINDOW_PICKER_COLS = 6

try:
    _sd0 = unique_drives.index(st.session_state.selected_drive)
except ValueError:
    _sd0 = 0
_chart_drive = st.selectbox(
    "Drive",
    options=unique_drives,
    index=_sd0,
    format_func=lambda x: (str(x)[:48] + "…") if len(str(x)) > 48 else str(x),
    key="chart_section_drive",
)
st.session_state.selected_drive = str(_chart_drive)
_idx_sel = drive_to_indices[_chart_drive]
if st.session_state.current_window not in _idx_sel:
    st.session_state.current_window = int(_idx_sel[0])
st.query_params["drive"] = str(_chart_drive)
st.query_params["w"] = str(int(st.session_state.current_window))
window_idx = int(st.session_state.current_window)

_n_w = len(_idx_sel)
_n_gdn_d = sum(1 for i in _idx_sel if gdn_window_flag[i])
dc1, dc2, dc3 = st.columns(3)
dc1.metric("Windows in this drive", f"{_n_w}")
dc2.metric("GDN alerts (this drive)", f"{_n_gdn_d}")
dc3.metric("GDN alerts (dataset)", f"{int(gdn_window_flag.sum()):,}")

st.markdown(f"##### {html.escape(str(_chart_drive))}")
fig_main = build_drive_traces_figure(
    _idx_sel, _n_w,
    chart_title=f"{str(_chart_drive)[:60]} — time → (each band = one window)",
)
st.plotly_chart(fig_main, use_container_width=True)

for _off in range(0, len(_idx_sel), WINDOW_PICKER_COLS):
    _chunk = _idx_sel[_off : _off + WINDOW_PICKER_COLS]
    _cols = st.columns(len(_chunk), gap="medium")
    for _ci, _gw in enumerate(_chunk):
        _k = _off + _ci + 1
        _alert = bool(gdn_window_flag[_gw])
        with _cols[_ci]:
            if st.button(
                f"Window {_k}",
                key=f"winpick_{_gw}",
                type="primary" if _alert else "secondary",
                help=f"Window {_k} in this drive (global id {_gw}). Opens Diagnostics.",
                use_container_width=True,
            ):
                st.session_state.current_window = int(_gw)
                st.session_state.selected_drive = str(_chart_drive)
                st.query_params["drive"] = str(_chart_drive)
                st.query_params["w"] = str(_gw)
                st.switch_page("pages/2_Diagnostics.py")
