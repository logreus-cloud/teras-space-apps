import os

# Shared cloud hosts report many CPUs; unbounded OpenMP threads make training crawl.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from teras.data import DATA_PATH, full_text, load_investigations
from teras.llm import is_configured, model_name, provider
from teras.retrieval import SearchIndex
from teras.ui import render_ask, render_attributes, render_compare, render_explore, render_rank
from teras.ui_predict import render_predict

st.set_page_config(page_title="Flame in Freefall — Teras", layout="wide")
load_dotenv()


@st.cache_data
def load_data(mtime_ns):
    return load_investigations()


@st.cache_resource
def load_index(items, mtime_ns):
    return SearchIndex.build(items)


mtime_ns = DATA_PATH.stat().st_mtime_ns
items = load_data(mtime_ns)
index = load_index(items, mtime_ns)
ready = is_configured()
st.title("Flame in Freefall")
st.caption("SI fire-safety insights from NASA PSI combustion data · Team Teras")

with st.sidebar:
    st.header("Teras")
    st.write("Explore microgravity combustion evidence for safer crewed missions.")
    if ready:
        st.success(f"SI provider: {provider().title()} ({model_name()})")
    else:
        st.warning("Add ANTHROPIC_API_KEY or GEMINI_API_KEY to .env")
    project_types = st.multiselect(
        "Project type", sorted({inv.project_type for inv in items if inv.project_type})
    )
    hardware = st.multiselect(
        "Hardware", sorted({inv.hardware_name for inv in items if inv.hardware_name})
    )
    query = st.text_input("Text search").strip().lower()

filtered = tuple(
    inv for inv in items
    if (not project_types or inv.project_type in project_types)
    and (not hardware or inv.hardware_name in hardware)
    and (not query or query in full_text(inv).lower())
)

tabs = st.tabs(["Explore", "Predict", "Rank by scenario", "Compare", "Ask", "Attributes"])
with tabs[0]:
    render_explore(items, filtered, ready)
with tabs[1]:
    render_predict(ready)
with tabs[2]:
    render_rank(items, ready)
with tabs[3]:
    render_compare(items, ready)
with tabs[4]:
    render_ask(index, ready)
with tabs[5]:
    render_attributes(items, ready)
