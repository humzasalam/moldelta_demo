"""MolDelta — AI-powered lead optimization demo.

Run with:
    streamlit run app.py
"""

import json
import os

import streamlit as st

# ── Page config MUST be first Streamlit call ──
st.set_page_config(
    page_title="MolDelta | Lead Optimization",
    page_icon="\U0001f9ec",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from utils.theme import inject_custom_css
from components.viz_panel import render_control_panel, render_viz_panel

# ── Apply Nord theme ──
inject_custom_css()


# ── Load data (cached) ──
@st.cache_data
def load_data():
    base = os.path.dirname(__file__)
    parents = []
    children_sets = []
    for i in [1, 2]:
        with open(os.path.join(base, "data", f"parent_{i}.json")) as f:
            parents.append(json.load(f))
        with open(os.path.join(base, "data", f"children_{i}.json")) as f:
            children = json.load(f)
            # Filter out molecules with null or zero binding_probability
            children = [
                child for child in children
                if child.get("binding_probability")
            ]
            children_sets.append(children)
    return parents, children_sets


parents, children_sets = load_data()


# ── Initialize session state ──
def init_session_state():
    defaults = {
        "flow_step": "WELCOME",
        "selected_parent_index": 0,
        "num_to_generate": 20,
        "parents_data": parents,
        "children_sets": children_sets,
        "parent_data": parents[0],
        "children_data": children_sets[0],
        "selected_molecule_id": None,
        "x_axis": "binding_probability",
        "y_axis": "hERG (nM)",
        "top_ids": [],  # Legacy key (kept for backward compat)
        "enumeration_completed": False,
        # Advanced optimization features
        "obj_props": [],
        "obj_dirs": {},
        "bp_guard_enabled": False,
        "bp_guard_value": 0.50,
        "parent_touched": False,
        "count_touched": False,
        "tracked_ids": set(),
        "pareto_ids": [],
        "topk_ids": [],
        "top_k": 1,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ── Two-column layout: controls left, chart right ──
ctrl_col, viz_col = st.columns([25, 75], gap="medium")

with ctrl_col:
    render_control_panel()

with viz_col:
    render_viz_panel()
