import pandas as pd
import streamlit as st

from . import cache
from .data import to_dataframe
from .llm import generate, generate_json
from .prompts import SYSTEM_EXPERT, attributes_prompt, compare_prompt, qa_prompt, rank_prompt, summary_prompt
from .validation import RANK_SCHEMA, normalize_attributes, normalize_rank


def _display(value):
    return ", ".join(value) if isinstance(value, tuple) else str(value) if value is not None else ""


def _call(kind, prompt, ready, json_mode=False, validator=None, schema=None):
    if not ready:
        st.info("Add ANTHROPIC_API_KEY or GEMINI_API_KEY to .env to enable SI features.")
        return None
    value = cache.get(kind, prompt)
    if value is not None:
        try:
            return validator(value) if validator else value
        except ValueError:
            pass
    try:
        value = generate_json(prompt, SYSTEM_EXPERT, schema) if json_mode else generate(prompt, SYSTEM_EXPERT)
        if validator:
            value = validator(value)
        cache.put(kind, prompt, value)
        return value
    except (RuntimeError, ValueError, OSError) as exc:
        st.error(str(exc))
        return None


def render_explore(items, filtered, ready):
    years = [year for inv in filtered for year in (inv.start_year, inv.end_year) if year]
    columns = st.columns(4)
    columns[0].metric("Investigations", len(filtered))
    columns[1].metric("Microgravity", sum(inv.project_type == "Microgravity Investigation" for inv in filtered))
    columns[2].metric("Ground", sum(inv.project_type == "Ground Investigation" for inv in filtered))
    columns[3].metric("Years", f"{min(years)}–{max(years)}" if years else "Unknown")
    st.dataframe(to_dataframe(filtered), width="stretch", hide_index=True)
    hardware = pd.Series([inv.hardware_name or "Unspecified" for inv in filtered]).value_counts()
    if not hardware.empty:
        st.bar_chart(hardware)
    if not filtered:
        st.info("No investigations match the filters.")
        return
    requested = st.query_params.get("investigation", "")
    options = [inv.accession for inv in filtered]
    index = options.index(requested) if requested in options else 0
    inv = st.selectbox("Investigation", filtered, index=index, format_func=lambda item: item.label)
    left, right = st.columns([1, 3])
    with left:
        if inv.thumbnail_url:
            try:
                st.image(inv.thumbnail_url)
            except Exception:
                st.caption("Thumbnail unavailable.")
    with right:
        st.subheader(inv.title or inv.accession)
        st.write(f"{inv.accession} · {inv.project_type} · {inv.platform}")
        st.markdown(f"[View on NASA PSI]({inv.page_url})")
        st.caption(f"Hardware: {inv.hardware_name or 'Unknown'} · Center: {inv.center or 'Unknown'}")
    for title, value in (("Objectives", inv.objectives), ("Approach", inv.approach), ("Hypothesis", inv.hypothesis)):
        with st.expander(title):
            st.write(value or "Not provided.")
    if inv.publications:
        st.write("Publications")
        for publication in inv.publications:
            st.write(f"- {publication}")
    if st.button("SI summary"):
        result = _call("summary", summary_prompt(inv), ready)
        if result is not None:
            st.markdown(result)


def render_rank(items, ready):
    examples = (
        "Fire in a lunar habitat at 34% O2 and 8.2 psi",
        "Smoke detection in ISS cabin",
        "Extinguishing a burning liquid fuel droplet",
    )
    columns = st.columns(3)
    for column, example in zip(columns, examples):
        if column.button(example):
            st.session_state.scenario = example
    scenario = st.text_area("Fire-safety scenario", key="scenario")
    lookup = {inv.accession: inv for inv in items}
    if st.button("Rank"):
        st.session_state.rank_results = None
        if not scenario.strip():
            st.info("Enter a scenario to rank investigations.")
        else:
            result = _call(
                "rank", rank_prompt(scenario, items), ready,
                json_mode=True, validator=lambda value: normalize_rank(value, lookup),
                schema=RANK_SCHEMA,
            )
            if result is not None:
                st.session_state.rank_results = {"scenario": scenario, "results": result}
    saved = st.session_state.get("rank_results")
    if not isinstance(saved, dict):
        return
    st.caption(f"Results for: {saved['scenario']}")
    for row in sorted(saved["results"], key=lambda item: item["score"], reverse=True):
        inv = lookup.get(row["accession"])
        if inv is None:
            continue
        score = row["score"]
        st.progress(score / 100, text=f"{inv.label} — {score}/100")
        st.write(row["reason"])
        st.markdown(f"[Open investigation](?investigation={inv.accession})")


def render_compare(items, ready):
    if len(items) < 2:
        st.info("At least two investigations are needed.")
        return
    a = st.selectbox("First investigation", items, key="compare_a", format_func=lambda inv: inv.label)
    b = st.selectbox("Second investigation", items, index=1, key="compare_b", format_func=lambda inv: inv.label)
    if a == b:
        st.info("Choose two different investigations.")
        return
    fields = (
        ("Title", "title"), ("Project type", "project_type"), ("Platform", "platform"),
        ("Hardware", "hardware_name"), ("Start year", "start_year"), ("End year", "end_year"),
        ("Keywords", "keywords"),
    )
    rows = [{"Field": "Accession", "Investigation A": a.accession, "Investigation B": b.accession}]
    for label, name in fields:
        rows.append({
            "Field": label,
            "Investigation A": _display(getattr(a, name)),
            "Investigation B": _display(getattr(b, name)),
        })
    st.table(pd.DataFrame(rows))
    if st.button("SI comparison"):
        result = _call("compare", compare_prompt(a, b), ready)
        if result is not None:
            st.markdown(result)


def render_ask(index, ready):
    for message in st.session_state.get("history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
            if message.get("sources"):
                st.markdown(message["sources"])
    question = st.chat_input("Ask about PSI combustion investigations")
    if not question:
        return
    matches = index.search(question, top_k=4)
    links = "Sources: " + ", ".join(
        f"[{inv.accession}](?investigation={inv.accession})" for inv, _ in matches
    )
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not matches:
            answer = "No matching PSI investigations were found."
        elif ready:
            answer = _call("qa", qa_prompt(question, [inv for inv, _ in matches]), ready)
        else:
            st.info("Add ANTHROPIC_API_KEY or GEMINI_API_KEY to .env to enable SI features.")
            answer = None
        if answer is not None:
            st.markdown(answer)
            if matches:
                st.markdown(links)
    old = st.session_state.get("history", [])
    updates = [{"role": "user", "text": question}]
    if answer is not None:
        updates.append({"role": "assistant", "text": answer, "sources": links if matches else ""})
    st.session_state.history = [*old, *updates]


def render_attributes(items, ready):
    if st.button("Extract attributes for all"):
        st.session_state.attributes = []
        records = []
        progress = st.progress(0)
        for number, inv in enumerate(items, 1):
            value = _call("attributes", attributes_prompt(inv), ready, True, normalize_attributes)
            if value is None:
                break
            records.append({
                "accession": inv.accession,
                "fuel_phase": value.get("fuel_phase", "unknown"),
                "fuels": ", ".join(value.get("fuels", [])) if isinstance(value.get("fuels"), list) else "",
                "gravity": value.get("gravity", "unknown"),
                "oxygen_conditions": value.get("oxygen_conditions", "unknown"),
                "fire_safety_area": ", ".join(value.get("fire_safety_area", []))
                if isinstance(value.get("fire_safety_area"), list) else "",
            })
            progress.progress(number / max(1, len(items)))
        if len(records) == len(items):
            st.session_state.attributes = records
    records = st.session_state.get("attributes", [])
    if not records:
        return
    table = pd.DataFrame(records)
    phases = st.multiselect("Fuel phase", sorted(table["fuel_phase"].dropna().unique()))
    areas = st.multiselect("Fire safety area", ["prevention", "detection", "suppression"])
    if phases:
        table = table[table["fuel_phase"].isin(phases)]
    if areas:
        table = table[table["fire_safety_area"].apply(lambda value: any(area in value for area in areas))]
    st.dataframe(table, width="stretch", hide_index=True)
