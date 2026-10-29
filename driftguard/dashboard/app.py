"""DriftGuard Streamlit Dashboard.

Run: ``streamlit run driftguard/dashboard/app.py``

A modern, high-tech, responsive MLOps dashboard for real-time data drift detection,
statistical monitoring, adaptive retraining, and validation-gated model promotion.
Includes live auto-reloader, streaming batch playback, transparent hero bar, and unified button ergonomics.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure repo root is on sys.path when invoked via streamlit run driftguard/dashboard/app.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import altair as alt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")

from driftguard.core.config import Settings
from driftguard.core.logging import configure_logging
from driftguard.data.simulation import SCENARIOS, DriftSimulator, ScenarioConfig
from driftguard.pipelines.pipeline import DriftGuardPipeline
from driftguard.pipelines.reporting import (
    METRICS, detector_frame, feature_frame, monitoring_frame, retrain_frame,
)

# ------------------------------------------------------------------ Page Config
st.set_page_config(
    page_title="DriftGuard — Live MLOps Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ High-End CSS Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    header[data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.85) !important;
        backdrop-filter: blur(8px) !important;
        height: 2.8rem !important;
        z-index: 100 !important;
    }

    .block-container {
        padding-top: 3.8rem !important;
        padding-bottom: 6rem !important;
        max-width: 1360px !important;
    }

    /* Floating Scroll to Top button */
    .floating-top-btn {
        position: fixed;
        bottom: 24px;
        right: 28px;
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-radius: 9999px;
        padding: 0.45rem 0.95rem;
        font-size: 0.82rem;
        font-weight: 700;
        color: #1e293b;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
        z-index: 9999;
        text-decoration: none;
        display: flex;
        align-items: center;
        gap: 0.35rem;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .floating-top-btn:hover {
        background: #2563eb;
        color: #ffffff !important;
        border-color: #2563eb;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35);
        transform: translateY(-2px);
    }

    /* Keyframe Animations */
    @keyframes pulse-glow-red {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
        70% { box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    @keyframes live-dot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.35; transform: scale(0.85); }
    }

    /* ------------------------------------------------------------------ Transparent Header Bar */
    .hero-banner-transparent {
        background: transparent;
        border: none;
        padding: 0.25rem 0 0.9rem 0;
        margin-bottom: 0.9rem;
        border-bottom: 1px solid #e2e8f0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .hero-left {
        display: flex;
        align-items: center;
        gap: 0.9rem;
    }

    .hero-logo-box-transparent {
        width: 48px;
        height: 48px;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.12) 0%, rgba(139, 92, 246, 0.12) 100%);
        border: 1.5px solid rgba(37, 99, 235, 0.25);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.65rem;
    }

    .hero-title-transparent {
        font-size: 1.55rem;
        font-weight: 850;
        letter-spacing: -0.03em;
        color: #0f172a;
        margin: 0;
        line-height: 1.15;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    .hero-subtitle-transparent {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 0.2rem;
        font-weight: 500;
        letter-spacing: 0.01em;
    }

    .hero-right-transparent {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        flex-wrap: wrap;
    }

    /* Live Pulsing Dot */
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        animation: live-dot 1.8s infinite ease-in-out;
    }

    .dot-green { background: #10b981; }
    .dot-red { background: #ef4444; }
    .dot-blue { background: #3b82f6; }

    /* Pill Badges adapted for light transparent background */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }

    .pill-green {
        background: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
    }

    .pill-red {
        background: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
        animation: pulse-glow-red 2.2s infinite;
    }

    .pill-amber {
        background: #fffbeb;
        color: #b45309;
        border: 1px solid #fde68a;
    }

    .pill-blue {
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
    }

    .pill-purple {
        background: #f5f3ff;
        color: #6d28d9;
        border: 1px solid #ddd6fe;
    }

    /* ------------------------------------------------------------------ Unified Button Styling */
    /* Forces ALL buttons to share the identical height, radius, font size, and transitions */
    div[data-testid="stButton"] > button,
    button[kind="primary"],
    button[kind="secondary"],
    .stButton > button {
        height: 42px !important;
        min-height: 42px !important;
        max-height: 42px !important;
        border-radius: 9px !important;
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.02em !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 1.1rem !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-sizing: border-box !important;
        line-height: 1 !important;
        white-space: nowrap !important;
    }

    div[data-testid="stButton"] > button[kind="primary"],
    button[kind="primary"] {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: 1px solid #1d4ed8 !important;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.22) !important;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover,
    button[kind="primary"]:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.32) !important;
        transform: translateY(-1px) !important;
    }

    div[data-testid="stButton"] > button[kind="secondary"],
    button[kind="secondary"] {
        background: #ffffff !important;
        color: #1e293b !important;
        border: 1.5px solid #cbd5e1 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    }

    div[data-testid="stButton"] > button[kind="secondary"]:hover,
    button[kind="secondary"]:hover {
        background: #f8fafc !important;
        border-color: #94a3b8 !important;
        color: #0f172a !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.06) !important;
    }

    /* Standardize inputs & selectboxes to 42px height for perfect alignment */
    div[data-baseweb="select"] > div {
        min-height: 42px !important;
        height: 42px !important;
        border-radius: 9px !important;
        border-color: #cbd5e1 !important;
    }

    div[data-testid="stNumberInput"] input {
        height: 42px !important;
        border-radius: 9px !important;
    }

    /* ------------------------------------------------------------------ Reloader Control Bar */
    .reloader-bar {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.65rem 1.15rem;
        margin-bottom: 1.25rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.9rem;
    }

    /* ------------------------------------------------------------------ KPI Metric Cards */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.15rem 1.35rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }

    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.07);
        border-color: #cbd5e1;
    }

    .kpi-top-bar {
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3.5px;
    }

    .kpi-label {
        font-size: 0.76rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 0.45rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .kpi-value {
        font-size: 1.95rem;
        font-weight: 900;
        color: #0f172a;
        line-height: 1.1;
        letter-spacing: -0.03em;
        font-family: 'JetBrains Mono', monospace;
    }

    .kpi-sub {
        font-size: 0.8rem;
        font-weight: 600;
        margin-top: 0.5rem;
        color: #64748b;
        display: flex;
        align-items: center;
        gap: 0.35rem;
    }

    /* Glass Cards */
    .glass-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02);
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 800;
        color: #0f172a;
        margin: 1.4rem 0 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: -0.015em;
    }

    /* Tab Customization */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #f1f5f9;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 1.1rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 42px;
        background-color: transparent;
        border-radius: 8px;
        color: #64748b;
        font-size: 0.88rem;
        font-weight: 600;
        padding: 0 18px;
        border: none;
        transition: all 0.18s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: #0f172a;
        background-color: rgba(255, 255, 255, 0.7);
    }

    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }

    code {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: #f1f5f9 !important;
        border: 1px solid #e2e8f0;
        border-radius: 4px;
        padding: 0.15rem 0.4rem;
        font-size: 0.85em;
        color: #2563eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ Pipeline Singleton & Session State
@st.cache_resource(show_spinner="Initializing DriftGuard MLOps Engine…")
def get_pipeline() -> DriftGuardPipeline:
    settings = Settings.from_env()
    configure_logging(settings.log_level)
    pipe = DriftGuardPipeline(settings)
    pipe.ensure_baseline()
    return pipe


pipe = get_pipeline()

# Session State Initialization
st.session_state.setdefault("live_stream_active", False)
st.session_state.setdefault("live_stream_idx", 0)
st.session_state.setdefault("live_stream_max", 12)
st.session_state.setdefault("auto_reload_enabled", False)
st.session_state.setdefault("auto_reload_sec", 3)
st.session_state.setdefault("last_reload_time", datetime.now().strftime("%H:%M:%S"))

# ------------------------------------------------------------------ Live Stream Engine Step
if st.session_state.get("live_stream_active", False):
    cur_idx = st.session_state["live_stream_idx"]
    max_idx = st.session_state["live_stream_max"]
    stream_scen = st.session_state.get("stream_scenario", "sudden")
    stream_onset = st.session_state.get("stream_onset", 4)
    stream_seed = st.session_state.get("stream_seed", 42)
    stream_retrain = st.session_state.get("stream_retrain", True)

    if cur_idx < max_idx:
        sim = DriftSimulator(
            pipe.splits,
            ScenarioConfig(name=stream_scen, drift_start=stream_onset, seed=stream_seed),
            pipe.settings.data.batch_size,
        )
        batch = sim.batch(cur_idx)
        step_res = pipe.process_batch(batch, allow_retrain=stream_retrain)
        st.session_state["live_stream_idx"] = cur_idx + 1
        st.session_state["last_reload_time"] = datetime.now().strftime("%H:%M:%S")
        action_tag = step_res.get("action", "none")
        if action_tag == "retrain_promoted":
            st.toast(f"⚡ Batch #{cur_idx}: Retrained & Promoted candidate to v{step_res.get('model_version_after')}!", icon="🏆")
        elif step_res.get("drift", {}).get("dataset_drift"):
            st.toast(f"🚨 Batch #{cur_idx}: Drift Detected ({step_res['drift']['n_flagged']}/3 detectors)!", icon="⚠️")
        else:
            st.toast(f"📥 Batch #{cur_idx}: Normal baseline batch processed.", icon="🟢")
    else:
        st.session_state["live_stream_active"] = False
        st.toast(f"🏁 Live stream finished ({max_idx} batches processed).", icon="🎉")

# ------------------------------------------------------------------ Read Pipeline State
events = pipe.history.all()
mon = monitoring_frame(events)
ret = retrain_frame(events)
summary = pipe.metrics_summary()
record = pipe.current_record()
latest_ev = summary["latest"]
counts = summary["counts"]
is_drifting = latest_ev and latest_ev["dataset_drift"]
deployed_v = f"v{record.version}"

# ------------------------------------------------------------------ Scenario Meta Dictionary
SCENARIO_META = {
    "baseline": {
        "title": "Normal Baseline (No Drift)",
        "badge": "STABLE",
        "desc": "Batches are sampled from reference distribution P(x). Model accuracy remains optimal.",
    },
    "covariate": {
        "title": "Covariate Shift",
        "badge": "P(x) SHIFT",
        "desc": "Input feature distribution P(x) tilts, but conditional label function P(y|x) is strictly preserved. Detectors trigger; accuracy may not decline.",
    },
    "sudden": {
        "title": "Sudden Feature Drift",
        "badge": "ABRUPT STEP",
        "desc": "A subset of critical features shifts instantaneously by 1.5σ at the onset batch, crossing decision boundaries and degrading F1.",
    },
    "gradual": {
        "title": "Gradual Drift Ramp",
        "badge": "RAMPED SHIFT",
        "desc": "1.5σ feature perturbation ramps up smoothly across 4 batches, simulating sensor degradation or seasonality.",
    },
    "concept": {
        "title": "Concept Drift (Label Shift)",
        "badge": "P(y|x) SHIFT",
        "desc": "Features x are untouched, but label mapping P(y|x) changes (label flips). Invisible to input-only detectors!",
    },
}

def render_empty(title: str, msg: str) -> None:
    st.markdown(
        f"""
        <div style="background:#f8fafc; border:1px dashed #cbd5e1; border-radius:14px; padding:2rem 1.5rem; text-align:center; margin:1rem 0;">
            <div style="font-size:1.8rem; margin-bottom:0.35rem;">📊</div>
            <div style="font-size:1.02rem; font-weight:700; color:#334155;">{title}</div>
            <div style="font-size:0.85rem; color:#64748b; margin-top:0.3rem; max-width:480px; margin-left:auto; margin-right:auto;">{msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ Sidebar: Controls & Live Simulation
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:0.6rem;">
            <div style="width:40px; height:40px; background:linear-gradient(135deg, #2563eb, #7c3aed); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.4rem;">🛡️</div>
            <div>
                <div style="font-size:1.2rem; font-weight:900; color:#0f172a; line-height:1.15;">DriftGuard</div>
                <div style="font-size:0.72rem; color:#64748b; font-weight:700; text-transform:uppercase; letter-spacing:0.05em;">Adaptive MLOps Engine</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;">
            <span style="font-size:0.84rem; font-weight:800; text-transform:uppercase; letter-spacing:0.05em; color:#1e293b;">⚡ Live Batch Streamer</span>
            <span class="pill pill-blue" style="font-size:0.65rem; padding:0.15rem 0.5rem;">Interactive</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    scen_choice = st.selectbox(
        "Simulation Scenario",
        SCENARIOS,
        index=2,
        format_func=lambda s: f"{s.capitalize()} — {SCENARIO_META[s]['badge']}",
        key="sim_scenario_select",
    )
    sm = SCENARIO_META[scen_choice]
    st.markdown(
        f"""
        <div style="background:#f8fafc; border-left:3px solid #3b82f6; border-radius:6px; padding:0.6rem 0.75rem; font-size:0.8rem; color:#334155; margin-bottom:0.75rem; line-height:1.4;">
            <strong>{sm['title']}</strong><br>{sm['desc']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        sim_batches = st.slider("Max Batches", 4, 30, 12, key="sim_batches_slider")
    with col_sim2:
        sim_onset = st.slider("Drift Onset", 0, 20, 4, key="sim_onset_slider")

    # Perfectly aligned Seed row with equal 50/50 columns
    col_sim3, col_sim4 = st.columns(2)
    with col_sim3:
        sim_seed = st.number_input("RNG Seed", 0, 999999, 42, step=1, key="sim_seed_input")
    with col_sim4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("🎲 Random", use_container_width=True, help="Generate random seed"):
            st.session_state["sim_seed_input"] = int(np.random.randint(0, 100000))
            st.rerun()

    sim_auto_retrain = st.checkbox("Adaptive Retraining Enabled", value=True, help="If checked, rule-based triggers train candidate models automatically.")

    # Equal size Play and Step buttons
    is_streaming = st.session_state.get("live_stream_active", False)
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if is_streaming:
            if st.button("⏸ Pause", type="secondary", use_container_width=True):
                st.session_state["live_stream_active"] = False
                st.rerun()
        else:
            if st.button("▶ Play Stream", type="primary", use_container_width=True):
                st.session_state["live_stream_active"] = True
                st.session_state["live_stream_idx"] = 0
                st.session_state["live_stream_max"] = sim_batches
                st.session_state["stream_scenario"] = scen_choice
                st.session_state["stream_onset"] = sim_onset
                st.session_state["stream_seed"] = int(sim_seed)
                st.session_state["stream_retrain"] = sim_auto_retrain
                st.rerun()

    with btn_col2:
        if st.button("⏩ Step Batch", type="secondary", use_container_width=True, help="Process exactly one streaming batch"):
            sim_step = DriftSimulator(
                pipe.splits,
                ScenarioConfig(name=scen_choice, drift_start=sim_onset, seed=int(sim_seed)),
                pipe.settings.data.batch_size,
            )
            step_idx = st.session_state.get("live_stream_idx", 0)
            b = sim_step.batch(step_idx)
            pipe.process_batch(b, allow_retrain=sim_auto_retrain)
            st.session_state["live_stream_idx"] = step_idx + 1
            st.session_state["last_reload_time"] = datetime.now().strftime("%H:%M:%S")
            st.rerun()

    # Progress indicator
    if is_streaming or st.session_state.get("live_stream_idx", 0) > 0:
        cur_p = min(st.session_state.get("live_stream_idx", 0), sim_batches)
        st.progress(cur_p / max(sim_batches, 1), text=f"Streaming: Batch {cur_p} of {sim_batches}")

    st.markdown("---")

    with st.expander("⚡ Bulk Scenario Ingest", expanded=False):
        st.caption("Process an entire scenario run at once.")
        if st.button("Run Entire Scenario", use_container_width=True):
            with st.spinner("Processing bulk scenario…"):
                evs = pipe.run_scenario(
                    ScenarioConfig(name=scen_choice, drift_start=sim_onset, seed=int(sim_seed)),
                    sim_batches,
                    sim_auto_retrain,
                )
            dup = sum(e["status"] == "duplicate_skipped" for e in evs)
            st.success(f"Processed {len(evs) - dup} batch(es)" + (f" ({dup} duplicate batches skipped)" if dup else ""))
            st.session_state["last_reload_time"] = datetime.now().strftime("%H:%M:%S")
            st.rerun()

    with st.expander("⚠️ System Reset", expanded=False):
        st.caption("Purge local registry and buffers back to a clean v1 baseline.")
        confirm = st.checkbox("Confirm State Wipe")
        if st.button("Reset System State", disabled=not confirm, use_container_width=True):
            pipe.reset()
            st.session_state["live_stream_active"] = False
            st.session_state["live_stream_idx"] = 0
            st.session_state["last_reload_time"] = datetime.now().strftime("%H:%M:%S")
            st.success("State reset successfully.")
            st.rerun()


# ------------------------------------------------------------------ Transparent Hero Bar
stream_tag = f"<span class='pill pill-blue'><span class='pulse-dot dot-blue'></span> STREAMING (B#{st.session_state.get('live_stream_idx', 0)})</span>" if is_streaming else ""
drift_pill_class = "pill-red" if is_drifting else "pill-green"
drift_dot_class = "dot-red" if is_drifting else "dot-green"
drift_label = "⚠️ DRIFT DETECTED" if is_drifting else "🟢 PIPELINE HEALTHY"
mlflow_label = "MLflow: Active" if pipe.tracker.enabled else "MLflow: Local SQLite"

# Clean single-line HTML string with ZERO 4-space markdown indentations
header_html = (
    f'<div id="top-anchor"></div>'
    f'<div class="hero-banner-transparent">'
    f'<div class="hero-left">'
    f'<div class="hero-logo-box-transparent">🛡️</div>'
    f'<div>'
    f'<div class="hero-title-transparent">DriftGuard Platform <span style="font-size:0.75rem; background:#f1f5f9; color:#475569; border:1px solid #e2e8f0; border-radius:6px; padding:0.15rem 0.5rem; font-weight:700;">v0.1.0</span></div>'
    f'<div class="hero-subtitle-transparent">Statistical Drift Detection · Model Health Monitoring · Adaptive Validation-Gated Retraining</div>'
    f'</div>'
    f'</div>'
    f'<div class="hero-right-transparent">'
    f'{stream_tag}'
    f'<span class="pill pill-blue">Active: {deployed_v}</span>'
    f'<span class="pill {drift_pill_class}"><span class="pulse-dot {drift_dot_class}"></span>{drift_label}</span>'
    f'<span class="pill pill-purple">{mlflow_label}</span>'
    f'<a href="#bottom-anchor" style="text-decoration:none;"><span class="pill" style="background:#f1f5f9; color:#475569; border:1px solid #cbd5e1; cursor:pointer;" title="Scroll to bottom">↓ Bottom</span></a>'
    f'</div>'
    f'</div>'
)
st.markdown(header_html, unsafe_allow_html=True)

# ------------------------------------------------------------------ Reloader Control Bar
col_rel_info, col_rel_ctrl = st.columns([1.6, 1.3])

with col_rel_info:
    auto_status = f"<span style='color:#10b981; font-weight:700;'>ON ({st.session_state.get('auto_reload_sec', 3)}s)</span>" if st.session_state.get("auto_reload_enabled", False) else "<span style='color:#64748b;'>OFF</span>"
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:1.2rem; font-size:0.85rem; color:#475569; padding-top:8px;">
            <div><strong>Telemetry Clock:</strong> <code style="color:#0f172a; font-weight:700;">{st.session_state.get('last_reload_time')}</code></div>
            <div><strong>Auto-Reload:</strong> {auto_status}</div>
            <div><strong>Batches Ingested:</strong> <strong style="color:#0f172a;">{counts['monitoring_runs']}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_rel_ctrl:
    r_col1, r_col2, r_col3 = st.columns([1.2, 1.1, 1])
    with r_col1:
        auto_toggle = st.toggle("Auto-Reload", value=st.session_state.get("auto_reload_enabled", False), key="auto_toggle_widget")
        if auto_toggle != st.session_state.get("auto_reload_enabled", False):
            st.session_state["auto_reload_enabled"] = auto_toggle
            st.rerun()
    with r_col2:
        interval = st.selectbox(
            "Interval",
            [2, 3, 5, 10, 30],
            index=[2, 3, 5, 10, 30].index(st.session_state.get("auto_reload_sec", 3)),
            format_func=lambda s: f"{s} sec",
            label_visibility="collapsed",
            key="interval_select_widget",
        )
        if interval != st.session_state.get("auto_reload_sec", 3):
            st.session_state["auto_reload_sec"] = interval
            st.rerun()
    with r_col3:
        if st.button("🔄 Sync", help="Manually re-sync state and refresh metrics", use_container_width=True):
            pipe.reload()
            st.session_state["last_reload_time"] = datetime.now().strftime("%H:%M:%S")
            st.rerun()

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ Main Navigation Tabs
tab_overview, tab_drift, tab_perf, tab_retrain, tab_registry, tab_research = st.tabs([
    "📊 Overview & Status",
    "🔍 Drift Monitoring",
    "📈 Model Performance",
    "🔄 Adaptive Retraining",
    "🗂️ Model Registry & MLflow",
    "📑 Research Benchmarks",
])

# ------------------------------------------------------------------ 1. OVERVIEW TAB
with tab_overview:
    # 4 Sleek Metric Cards
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-bar" style="background:linear-gradient(90deg, #2563eb, #60a5fa);"></div>
                <div class="kpi-label">Deployed Model <span>📦</span></div>
                <div class="kpi-value">{deployed_v}</div>
                <div class="kpi-sub">
                    Status: <strong style="color:#059669; margin-left:3px;">{record.status.upper()}</strong> · {record.train_rows} train rows
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        if latest_ev is None:
            drift_val = "IDLE"
            drift_sub = "Awaiting streaming batch"
            drift_color = "#64748b"
            accent = "#94a3b8"
        else:
            n_flags = latest_ev["n_flagged"]
            drift_val = "DRIFT" if is_drifting else "STABLE"
            drift_color = "#dc2626" if is_drifting else "#059669"
            drift_sub = f"{n_flags} / 3 detectors triggered"
            accent = "#ef4444" if is_drifting else "#10b981"

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-bar" style="background:{accent};"></div>
                <div class="kpi-label">Drift Status <span>⚡</span></div>
                <div class="kpi-value" style="color:{drift_color};">{drift_val}</div>
                <div class="kpi-sub">{drift_sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        if latest_ev is None or not latest_ev.get("performance"):
            f1_str = "–"
            delta_str = "No labeled batches"
            accent = "#cbd5e1"
        else:
            cur_f1 = latest_ev["performance"].get("f1")
            base_f1 = summary["baseline_reference_metrics"].get("f1", 0.0)
            f1_str = f"{cur_f1:.3f}" if cur_f1 is not None else "–"
            diff = (cur_f1 - base_f1) if cur_f1 is not None else 0.0
            diff_color = "#059669" if diff >= 0 else "#dc2626"
            delta_str = f"<span style='color:{diff_color}; font-weight:700;'>{diff:+.3f}</span> vs baseline ({base_f1:.3f})"
            accent = "#10b981" if diff >= -0.05 else "#ef4444"

        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-bar" style="background:{accent};"></div>
                <div class="kpi-label">Latest Batch F1 <span>🎯</span></div>
                <div class="kpi-value">{f1_str}</div>
                <div class="kpi-sub">{delta_str}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c4:
        promotions = counts["promoted"]
        attempts = counts["retrain_attempts"]
        ratio_str = f"{(promotions/attempts*100):.0f}%" if attempts else "–"
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-top-bar" style="background:linear-gradient(90deg, #7c3aed, #a855f7);"></div>
                <div class="kpi-label">Retrain & Promotion <span>🔄</span></div>
                <div class="kpi-value">{promotions} <span style="font-size:1.15rem; color:#64748b; font-weight:500;">/ {attempts}</span></div>
                <div class="kpi-sub">
                    Promoted ratio: <strong>{ratio_str}</strong> of attempts
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 10px'></div>", unsafe_allow_html=True)

    # State Machine Flow
    st.markdown(
        """
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.9rem;">
                <span style="font-size:0.92rem; font-weight:800; color:#1e293b; text-transform:uppercase; letter-spacing:0.05em;">
                    DriftGuard Closed-Loop Operational State Machine
                </span>
                <span class="pill pill-green">Synchronized</span>
            </div>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(185px, 1fr)); gap:0.8rem; text-align:center;">
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:0.9rem 0.65rem;">
                    <div style="font-size:1.35rem;">📥</div>
                    <div style="font-weight:800; font-size:0.85rem; color:#0f172a; margin-top:0.25rem;">1. Stream Ingest</div>
                    <div style="font-size:0.75rem; color:#64748b;">Schema & Hash Verification</div>
                </div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:0.9rem 0.65rem;">
                    <div style="font-size:1.35rem;">🔬</div>
                    <div style="font-weight:800; font-size:0.85rem; color:#0f172a; margin-top:0.25rem;">2. Drift Detectors</div>
                    <div style="font-size:0.75rem; color:#64748b;">KS · PSI · Domain Classifier</div>
                </div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:0.9rem 0.65rem;">
                    <div style="font-size:1.35rem;">⚖️</div>
                    <div style="font-weight:800; font-size:0.85rem; color:#0f172a; margin-top:0.25rem;">3. Trigger Rules</div>
                    <div style="font-size:0.75rem; color:#64748b;">Consecutive Drift · Cooldown</div>
                </div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:0.9rem 0.65rem;">
                    <div style="font-size:1.35rem;">🛡️</div>
                    <div style="font-weight:800; font-size:0.85rem; color:#0f172a; margin-top:0.25rem;">4. Promotion Gate</div>
                    <div style="font-size:0.75rem; color:#64748b;">Holdout Split & Anchor Recall</div>
                </div>
                <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:0.9rem 0.65rem;">
                    <div style="font-size:1.35rem;">🚀</div>
                    <div style="font-weight:800; font-size:0.85rem; color:#0f172a; margin-top:0.25rem;">5. Atomic Deploy</div>
                    <div style="font-size:0.75rem; color:#64748b;">Pointer Swap & Ref Refresh</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Activity Stream
    st.markdown("<div class='section-title'>📋 Real-Time Activity Feed</div>", unsafe_allow_html=True)
    if mon.empty:
        render_empty("No Batches Logged Yet", "Start a live stream or click 'Step Batch' in the sidebar to feed incoming data into the pipeline.")
    else:
        stream_df = mon.tail(15).iloc[::-1].copy()
        stream_df["Status"] = stream_df["dataset_drift"].apply(lambda d: "🚨 DRIFT" if d else "✅ STABLE")
        stream_df["Detectors Flagged"] = stream_df["n_flagged"].apply(lambda n: f"{n} / 3")
        stream_df["F1 Score"] = stream_df["f1"].apply(lambda f: f"{f:.3f}" if pd.notna(f) else "–")
        stream_df["Model"] = "v" + stream_df["model_version"].astype(str)
        stream_df["Action"] = stream_df["action"].apply(
            lambda a: "🔄 Retrained & Promoted" if a == "retrain_promoted" else ("⚠️ Retrain Rejected" if a == "retrain_rejected" else "None")
        )

        display_cols = ["batch_seq", "scenario", "batch_index", "Model", "Detectors Flagged", "Status", "F1 Score", "Action"]
        st.dataframe(
            stream_df[display_cols].rename(columns={"batch_seq": "Seq #", "scenario": "Scenario", "batch_index": "Index"}),
            use_container_width=True,
            hide_index=True,
        )


# ------------------------------------------------------------------ 2. DRIFT MONITORING TAB
with tab_drift:
    ok_events = [e for e in events if e.get("event_type") == "monitoring" and e.get("status") == "ok"]
    if not ok_events:
        render_empty("No Drift Data Collected", "Run a scenario in the sidebar to stream batches and inspect statistical drift distributions.")
    else:
        col_b1, col_b2 = st.columns([2.5, 1])
        with col_b1:
            labels = [f"Batch #{e['batch_seq']} — {e['scenario'].capitalize()} (Idx {e['batch_index']}) · {'🚨 DRIFT' if e['drift']['dataset_drift'] else '✅ OK'}" for e in ok_events]
            pick = st.selectbox("Inspect Streaming Batch:", range(len(ok_events)), index=len(ok_events) - 1, format_func=lambda i: labels[i])
        with col_b2:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            ev = ok_events[pick]
            st.markdown(
                f"<div style='text-align:right;'><span class='pill {'pill-red' if ev['drift']['dataset_drift'] else 'pill-green'}'>Dataset Drift: {'FLAGGED' if ev['drift']['dataset_drift'] else 'NOT DETECTED'}</span></div>",
                unsafe_allow_html=True,
            )

        # 4 Detector Cards
        df_det = detector_frame(ev)
        d_cols = st.columns(4)

        ks_row = df_det[df_det["detector"] == "ks"]
        psi_row = df_det[df_det["detector"] == "psi"]
        dom_row = df_det[df_det["detector"] == "domain_classifier"]
        pred_drift = ev.get("prediction_drift", {})

        with d_cols[0]:
            is_ks = not ks_row.empty and bool(ks_row.iloc[0]["drifted"])
            ks_score = ks_row.iloc[0]["score"] if not ks_row.empty else 0.0
            ks_n = ks_row.iloc[0]["features_flagged"] if not ks_row.empty else 0
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-top-bar" style="background:{'#ef4444' if is_ks else '#10b981'};"></div>
                    <div class="kpi-label">Kolmogorov-Smirnov (KS)</div>
                    <div class="kpi-value" style="font-size:1.6rem; color:{'#dc2626' if is_ks else '#059669'};">
                        {ks_score:.1%} <span style="font-size:0.8rem; color:#64748b; font-weight:500;">(Thresh 20%)</span>
                    </div>
                    <div class="kpi-sub">
                        Flagged features: <strong>{ks_n} / 30</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with d_cols[1]:
            is_psi = not psi_row.empty and bool(psi_row.iloc[0]["drifted"])
            psi_score = psi_row.iloc[0]["score"] if not psi_row.empty else 0.0
            psi_n = psi_row.iloc[0]["features_flagged"] if not psi_row.empty else 0
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-top-bar" style="background:{'#ef4444' if is_psi else '#10b981'};"></div>
                    <div class="kpi-label">Population Stability (PSI)</div>
                    <div class="kpi-value" style="font-size:1.6rem; color:{'#dc2626' if is_psi else '#059669'};">
                        {psi_score:.1%} <span style="font-size:0.8rem; color:#64748b; font-weight:500;">(Thresh 20%)</span>
                    </div>
                    <div class="kpi-sub">
                        Flagged features: <strong>{psi_n} / 30</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with d_cols[2]:
            is_dom = not dom_row.empty and bool(dom_row.iloc[0]["drifted"])
            dom_score = dom_row.iloc[0]["score"] if not dom_row.empty else 0.5
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-top-bar" style="background:{'#ef4444' if is_dom else '#10b981'};"></div>
                    <div class="kpi-label">Domain Classifier</div>
                    <div class="kpi-value" style="font-size:1.6rem; color:{'#dc2626' if is_dom else '#059669'};">
                        {dom_score:.3f} <span style="font-size:0.8rem; color:#64748b; font-weight:500;">(AUC ≥ 0.65)</span>
                    </div>
                    <div class="kpi-sub">
                        5-Fold CV Linear AUC
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with d_cols[3]:
            is_pred = bool(pred_drift.get("drifted", False))
            p_val = pred_drift.get("p_value", 1.0)
            r_rate = pred_drift.get("ref_positive_rate", 0.0)
            c_rate = pred_drift.get("cur_positive_rate", 0.0)
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-top-bar" style="background:{'#ef4444' if is_pred else '#10b981'};"></div>
                    <div class="kpi-label">Prediction Shift</div>
                    <div class="kpi-value" style="font-size:1.6rem; color:{'#dc2626' if is_pred else '#059669'};">
                        {'DRIFT' if is_pred else 'STABLE'} <span style="font-size:0.8rem; color:#64748b; font-weight:500;">(p={p_val:.2g})</span>
                    </div>
                    <div class="kpi-sub">
                        Rate: <strong>{r_rate:.0%} ➔ {c_rate:.0%}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)

        # Multi-Detector Timeline Chart
        st.markdown("<div class='section-title'>📈 Detector Scores Timeline</div>", unsafe_allow_html=True)

        score_cols = [c for c in mon.columns if c.startswith("score_")]
        if score_cols:
            melted_scores = mon.melt(
                id_vars=["batch_seq", "scenario"],
                value_vars=score_cols,
                var_name="detector",
                value_name="score",
            )
            name_map = {
                "score_ks": "KS Test (Drifted Feature Share)",
                "score_psi": "PSI (Drifted Feature Share)",
                "score_domain_classifier": "Domain Classifier (CV-AUC)",
            }
            melted_scores["Detector"] = melted_scores["detector"].map(lambda d: name_map.get(d, d))

            score_chart = alt.Chart(melted_scores).mark_line(point=alt.OverlayMarkDef(filled=True, size=55)).encode(
                x=alt.X("batch_seq:Q", title="Streaming Batch Sequence Number"),
                y=alt.Y("score:Q", title="Score / AUC", scale=alt.Scale(domain=[0, 1.05])),
                color=alt.Color("Detector:N", scale=alt.Scale(range=["#2563eb", "#8b5cf6", "#f59e0b"])),
                tooltip=["batch_seq:Q", "scenario:N", "Detector:N", alt.Tooltip("score:Q", format=".3f")],
            ).properties(height=320)

            thresh_df = pd.DataFrame([
                {"threshold": 0.20, "label": "Feature Drift Threshold (20%)"},
                {"threshold": 0.65, "label": "Domain Classifier AUC (0.65)"},
            ])
            thresh_rule = alt.Chart(thresh_df).mark_rule(strokeDash=[4, 4], color="#94a3b8", opacity=0.8).encode(
                y="threshold:Q",
                tooltip=["label:N", "threshold:Q"],
            )

            st.altair_chart(score_chart + thresh_rule, use_container_width=True)

        # Top Drifted Features & Distribution Inspector
        col_f1, col_f2 = st.columns([1, 1])

        with col_f1:
            st.markdown("<div class='section-title'>📊 Top Drifted Features (Ranked by PSI)</div>", unsafe_allow_html=True)
            ff = feature_frame(ev)
            if ff.empty or "psi_score" not in ff:
                st.caption("No feature-level PSI metrics available for this batch.")
            else:
                top_ff = ff.sort_values("psi_score", ascending=False).head(10).copy()
                top_ff["is_drifted"] = top_ff.get("psi_flag", top_ff["psi_score"] > 0.25)

                bar_chart = alt.Chart(top_ff).mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5).encode(
                    y=alt.Y("feature:N", sort="-x", title=None),
                    x=alt.X("psi_score:Q", title="Population Stability Index (PSI)"),
                    color=alt.Color(
                        "is_drifted:N",
                        scale=alt.Scale(domain=[True, False], range=["#ef4444", "#94a3b8"]),
                        legend=alt.Legend(title="Drift Flagged"),
                    ),
                    tooltip=["feature:N", alt.Tooltip("psi_score:Q", format=".3f"), "is_drifted:N"],
                ).properties(height=340)

                st.altair_chart(bar_chart, use_container_width=True)

        with col_f2:
            st.markdown("<div class='section-title'>🔬 Reference vs Batch Feature Distribution</div>", unsafe_allow_html=True)
            lb = pipe.last_batch()
            if lb is None:
                st.caption("No batch buffer currently stored.")
            else:
                Xb, _, meta = lb
                default_feat = "worst perimeter" if "worst perimeter" in pipe.features else pipe.features[0]
                feat = st.selectbox("Inspect Feature:", pipe.features, index=pipe.features.index(default_feat))

                fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=120)
                fig.patch.set_facecolor("#ffffff")
                ax.set_facecolor("#fafbfc")

                ref_vals = pipe.reference[feat].dropna().values
                cur_vals = Xb[feat].dropna().values

                bins = np.histogram_bin_edges(np.r_[ref_vals, cur_vals], bins=24)
                ax.hist(ref_vals, bins=bins, alpha=0.45, density=True, color="#2563eb", label="Reference Distribution (P(x))", edgecolor="#1d4ed8")
                ax.hist(cur_vals, bins=bins, alpha=0.55, density=True, color="#ea580c", label=f"Batch #{meta.get('batch_seq', '?')} Incoming", edgecolor="#c2410c")

                ax.axvline(np.mean(ref_vals), color="#1d4ed8", linestyle="--", linewidth=1.5, label="Ref Mean")
                ax.axvline(np.mean(cur_vals), color="#c2410c", linestyle="--", linewidth=1.5, label="Batch Mean")

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_color("#cbd5e1")
                ax.spines["bottom"].set_color("#cbd5e1")
                ax.set_xlabel(feat, fontsize=9.5, fontweight="600", color="#334155")
                ax.set_ylabel("Probability Density", fontsize=9.5, fontweight="600", color="#334155")
                ax.tick_params(colors="#64748b", labelsize=8.5)
                ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#e2e8f0", fontsize=8)

                st.pyplot(fig)
                plt.close(fig)


# ------------------------------------------------------------------ 3. MODEL PERFORMANCE TAB
with tab_perf:
    lab = mon[mon["labeled"] == True] if not mon.empty else mon  # noqa: E712
    if lab.empty:
        render_empty("Ground-Truth Labels Required", "Performance metrics (F1, Accuracy, Recall) require ground truth labels from the batch stream.")
    else:
        st.markdown("<div class='section-title'>🎯 Deployed Model Metrics Across Stream</div>", unsafe_allow_html=True)
        selected_metrics = st.multiselect("Select Performance Metrics to Track:", list(METRICS), default=["f1", "recall", "precision"])

        if selected_metrics:
            perf_melted = lab.melt(
                id_vars=["batch_seq", "scenario", "model_version"],
                value_vars=selected_metrics,
                var_name="metric",
                value_name="score",
            )
            perf_melted["Model"] = "v" + perf_melted["model_version"].astype(str)

            perf_chart = alt.Chart(perf_melted).mark_line(point=alt.OverlayMarkDef(filled=True, size=55)).encode(
                x=alt.X("batch_seq:Q", title="Streaming Batch Sequence"),
                y=alt.Y("score:Q", title="Metric Score", scale=alt.Scale(domain=[0.2, 1.02])),
                color=alt.Color("metric:N", scale=alt.Scale(range=["#10b981", "#3b82f6", "#8b5cf6", "#f59e0b", "#ec4899"])),
                tooltip=["batch_seq:Q", "scenario:N", "Model:N", "metric:N", alt.Tooltip("score:Q", format=".3f")],
            ).properties(height=320)

            base_f1 = summary["baseline_reference_metrics"].get("f1", 0.965)
            base_rule = alt.Chart(pd.DataFrame([{"base": base_f1}])).mark_rule(
                strokeDash=[5, 5], color="#059669", opacity=0.85
            ).encode(y="base:Q", tooltip=[alt.Tooltip("base:Q", title="Baseline Reference F1")])

            st.altair_chart(perf_chart + base_rule, use_container_width=True)

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.markdown("<div class='section-title'>📊 Average Performance by Drift Scenario</div>", unsafe_allow_html=True)
            scen_group = lab.groupby("scenario")[list(METRICS)].mean().round(3).reset_index()
            scen_melted = scen_group.melt(id_vars=["scenario"], value_vars=["f1", "recall", "precision"], var_name="Metric", value_name="Score")
            scen_bar = alt.Chart(scen_melted).mark_bar().encode(
                x=alt.X("scenario:N", title=None),
                y=alt.Y("Score:Q", title="Mean Score"),
                color=alt.Color("Metric:N", scale=alt.Scale(range=["#10b981", "#3b82f6", "#8b5cf6"])),
                xOffset="Metric:N",
                tooltip=["scenario:N", "Metric:N", alt.Tooltip("Score:Q", format=".3f")],
            ).properties(height=280)

            st.altair_chart(scen_bar, use_container_width=True)

        with col_p2:
            st.markdown("<div class='section-title'>💡 The Drift vs Degradation Principle</div>", unsafe_allow_html=True)
            st.markdown(
                """
                <div class="glass-card" style="background:#f8fafc; border-left:4px solid #3b82f6; font-size:0.88rem; line-height:1.55;">
                    <strong style="color:#1e3a8a;">Core Empirical Finding:</strong>
                    <ul style="margin-top:0.4rem; padding-left:1.2rem; color:#334155;">
                        <li><strong>Feature Drift ≠ Degradation:</strong> Under <em>covariate shift</em>, input distributions change and trigger drift alarms, but model decision boundaries remain accurate.</li>
                        <li><strong>Degradation ≠ Feature Drift:</strong> Under <em>concept drift</em>, input features look completely normal, but true labels shift—causing model F1 to collapse without triggering input drift detectors.</li>
                    </ul>
                    <div style="font-size:0.8rem; color:#64748b; margin-top:0.5rem;">
                        This validates DriftGuard's multi-tier trigger policy: monitoring both inputs and downstream predictions.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ------------------------------------------------------------------ 4. RETRAINING & PROMOTION TAB
with tab_retrain:
    cfg = pipe.settings.retrain
    st.markdown(
        f"""
        <div class="glass-card" style="background:#f8fafc; border-left:4px solid #8b5cf6;">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem;">
                <div>
                    <strong style="font-size:0.95rem; color:#1e293b;">Active Retraining Trigger Policy:</strong>
                    <code style="font-size:0.9rem; font-weight:700; color:#4338ca; margin-left:0.3rem;">{cfg.policy.upper()}</code>
                </div>
                <div style="font-size:0.8rem; color:#64748b;">
                    Requires <strong>{cfg.min_detectors_drifted}</strong> detectors OR F1 drop ≥ <strong>{cfg.max_metric_drop:.2f}</strong> for <strong>{cfg.min_consecutive}</strong> consecutive batches
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_r1, col_r2 = st.columns([3, 1.2])
    with col_r1:
        st.markdown("<div class='section-title'>🔄 Retraining Event History</div>", unsafe_allow_html=True)
    with col_r2:
        reason_txt = st.text_input("Manual Retrain Reason", value="Manual dashboard trigger", label_visibility="collapsed")
        if st.button("⚡ Force Retrain Now", type="secondary", use_container_width=True):
            with st.spinner("Training and validating candidate model…"):
                res = pipe.retrain(force=True, reason=reason_txt)
            if res["status"] == "skipped":
                st.warning(f"Retrain skipped: {res.get('message', 'Window already consumed')}")
            else:
                st.success(f"Candidate v{res['candidate_version']} was {res['status'].upper()}!")
                st.rerun()

    if ret.empty:
        render_empty("No Retraining Events Yet", "Run a drifting scenario or click 'Force Retrain Now' to trigger a candidate evaluation.")
    else:
        st.dataframe(
            ret.iloc[::-1][["timestamp", "status", "forced", "previous_version", "candidate_version", "current_holdout_f1", "candidate_holdout_f1", "decision"]].rename(
                columns={
                    "timestamp": "Timestamp",
                    "status": "Outcome",
                    "forced": "Forced?",
                    "previous_version": "Deployed v",
                    "candidate_version": "Candidate v",
                    "current_holdout_f1": "Deployed F1",
                    "candidate_holdout_f1": "Candidate F1",
                    "decision": "Reasons / Gate Decision",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    done = [e for e in events if e.get("event_type") == "retrain" and e.get("decision")]
    if done:
        st.markdown("<div class='section-title'>🛡️ Promotion Gate Validation Inspector</div>", unsafe_allow_html=True)
        ret_labels = [f"Retrain Event #{i+1} — Candidate v{e['candidate_version']} ({e['status'].upper()}) · Deployed v{e['previous_version']}" for i, e in enumerate(done)]
        pick_ret = st.selectbox("Select Retrain Event to Audit:", range(len(done)), index=len(done) - 1, format_func=lambda i: ret_labels[i])
        audit_e = done[pick_ret]

        dec = audit_e["decision"]
        is_promoted = dec.get("promote", False)

        st.markdown(
            f"""
            <div class="glass-card" style="border-left: 4px solid {'#10b981' if is_promoted else '#ef4444'};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:1.1rem; font-weight:700; color:#0f172a;">
                            Candidate Model v{audit_e['candidate_version']} — {'🏆 PROMOTED TO PRODUCTION' if is_promoted else '⛔ PROMOTION REJECTED'}
                        </span>
                        <div style="font-size:0.85rem; color:#64748b; margin-top:0.2rem;">
                            Trigger Reason: <strong>{(' | '.join(audit_e.get('trigger_reasons', [])) or 'Manual')}</strong>
                        </div>
                    </div>
                    <span class="pill {'pill-green' if is_promoted else 'pill-red'}">
                        {audit_e['status'].upper()}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_g1, col_g2 = st.columns([1, 1])

        with col_g1:
            st.markdown("<strong>Gate Checklist Breakdown</strong>", unsafe_allow_html=True)
            gate_rows = []
            for check_name, check_data in dec.get("checks", {}).items():
                gate_rows.append({
                    "Gate Check": check_name.replace("_", " ").title(),
                    "Result": "✅ PASSED" if check_data.get("passed") else "❌ FAILED",
                    "Measured Value": str(round(check_data.get("value"), 4)) if isinstance(check_data.get("value"), (int, float)) else str(check_data.get("value")),
                    "Required Threshold": str(check_data.get("threshold")),
                })
            st.dataframe(pd.DataFrame(gate_rows), use_container_width=True, hide_index=True)

        with col_g2:
            st.markdown("<strong>Candidate vs Deployed Holdout Metrics</strong>", unsafe_allow_html=True)
            cm = audit_e.get("candidate_metrics", {}).get("holdout", {})
            um = audit_e.get("current_metrics", {}).get("holdout", {})
            if cm and um:
                comp_df = pd.DataFrame({"Deployed": um, "Candidate": cm}).drop(index=["n", "positive_rate"], errors="ignore")
                comp_df["Delta"] = comp_df["Candidate"] - comp_df["Deployed"]
                st.dataframe(comp_df.round(3), use_container_width=True)


# ------------------------------------------------------------------ 5. MODEL REGISTRY & MLFLOW TAB
with tab_registry:
    st.markdown("<div class='section-title'>🗂️ Model Registry Version Lineage</div>", unsafe_allow_html=True)
    recs = pipe.registry.list()

    reg_rows = []
    for r in recs:
        is_cur = r.version == record.version
        reg_rows.append({
            "Version": f"v{r.version}" + (" (Deployed ⭐)" if is_cur else ""),
            "Status": r.status.upper(),
            "Created At": r.created_at[:19].replace("T", " "),
            "Parent": f"v{r.parent_version}" if r.parent_version else "–",
            "Training Rows": r.train_rows,
            "Holdout F1": f"{(r.reference_metrics or {}).get('f1', 0.0):.3f}",
            "MLflow Run ID": r.mlflow_run_id or "–",
            "Trigger Reason": r.trigger_reason,
        })
    st.dataframe(pd.DataFrame(reg_rows), use_container_width=True, hide_index=True)

    with st.expander("🔍 Deployed Model Architecture & Hyperparameters"):
        st.json(record.params)

    st.markdown("<div class='section-title'>🧪 MLflow Experiment Tracking Runs</div>", unsafe_allow_html=True)
    runs = pipe.tracker.runs_frame()
    if runs is None or runs.empty:
        render_empty("MLflow Tracking Inactive / No Runs Logged", "MLflow SQLite store is ready. Runs log automatically during baseline training and monitoring.")
    else:
        keep = [c for c in runs.columns if c in ("run_id", "start_time", "tags.mlflow.runName", "tags.driftguard.kind", "tags.driftguard.decision", "tags.driftguard.action")]
        st.dataframe(runs[keep].rename(columns={
            "run_id": "Run ID",
            "start_time": "Timestamp",
            "tags.mlflow.runName": "Run Name",
            "tags.driftguard.kind": "Kind",
            "tags.driftguard.decision": "Decision",
            "tags.driftguard.action": "Action",
        }), use_container_width=True, hide_index=True)


# ------------------------------------------------------------------ 6. RESEARCH BENCHMARKS TAB
with tab_research:
    exp = pipe.paths.experiments_dir
    quick_exp = _REPO_ROOT / "artifacts" / "experiments_quick"
    target_exp = exp if (exp / "RESULTS.md").exists() else quick_exp

    if not (target_exp / "RESULTS.md").exists():
        render_empty(
            "No Experiment Results Found",
            "Run <code>python -m driftguard.experiments.run_all --quick</code> to generate research benchmark tables and plots.",
        )
    else:
        st.markdown(
            f"""
            <div class="glass-card" style="background:#f8fafc; border-left:4px solid #2563eb;">
                <div style="font-weight:700; color:#0f172a; font-size:1.05rem;">
                    Empirical Research Results & Reproducibility Suite
                </div>
                <div style="font-size:0.85rem; color:#64748b; margin-top:0.25rem;">
                    Loaded from <code>{target_exp.name}</code>. Every table and figure below was evaluated across controlled seeds and synthetic shift scenarios.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        for name, title in [
            ("detection_fpr_pooled.csv", "1. False-Positive Rates (No-Drift Baseline Batches)"),
            ("detection_summary.csv", "2. Detector Power & Detection Delay by Scenario"),
            ("static_performance_shift.csv", "3. Static Model Degradation Before vs After Drift"),
            ("sensitivity_summary.csv", "4. Sensitivity to Shift Magnitude (0.1σ to 1.5σ)"),
            ("retraining_summary.csv", "5. Retraining Policy Arms Evaluation"),
        ]:
            f = target_exp / name
            if f.exists():
                with st.expander(title, expanded=name.startswith("detection_fpr")):
                    st.dataframe(pd.read_csv(f).round(3), use_container_width=True, hide_index=True)

        st.markdown("<div class='section-title'>🖼️ Generated Research Visualizations</div>", unsafe_allow_html=True)
        img_cols = st.columns(2)
        images = [
            ("fig_detection_rates.png", "Detector Power Across Scenarios"),
            ("fig_detection_delay.png", "Mean Detection Delay in Batches"),
            ("fig_static_f1_timeline.png", "Static Model F1 Degradation Timeline"),
            ("fig_sensitivity.png", "Sensitivity Curve vs Shift Magnitude"),
            ("fig_retraining_f1_timeline.png", "Retraining Arms Recovery Comparison"),
            ("fig_retraining_outcomes.png", "Promotion vs Rejection Ratios"),
        ]
        col_idx = 0
        for img_file, img_caption in images:
            img_path = target_exp / img_file
            if img_path.exists():
                with img_cols[col_idx % 2]:
                    st.image(str(img_path), caption=img_caption, use_column_width=True)
                col_idx += 1

# ------------------------------------------------------------------ Bottom Anchor & Footer Navigation
st.markdown(
    """
    <div id="bottom-anchor"></div>
    <div style="margin-top: 3.5rem; padding-top: 1.5rem; border-top: 1px solid #e2e8f0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem;">
        <div style="font-size: 0.82rem; color: #64748b;">
            🛡️ <strong>DriftGuard v0.1.0</strong> · Automated MLOps Adaptive Monitoring Framework · Python 3.12
        </div>
        <div style="display: flex; gap: 0.75rem;">
            <a href="#top-anchor" style="text-decoration: none;">
                <button style="height: 38px; padding: 0 1.1rem; border-radius: 9px; border: 1.5px solid #cbd5e1; background: #ffffff; color: #1e293b; font-weight: 600; font-size: 0.84rem; cursor: pointer; display: inline-flex; align-items: center; gap: 0.4rem; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
                    ↑ Back to Top
                </button>
            </a>
        </div>
    </div>
    <a href="#top-anchor" class="floating-top-btn" title="Back to Top">
        <span>↑</span> <span>Top</span>
    </a>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ 7. Live Reloader Tick (Bottom of script)
if st.session_state.get("live_stream_active", False):
    time.sleep(1.8)
    st.rerun()
elif st.session_state.get("auto_reload_enabled", False):
    sec = st.session_state.get("auto_reload_sec", 3)
    time.sleep(sec)
    st.rerun()
