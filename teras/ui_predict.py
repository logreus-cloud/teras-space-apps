import pandas as pd
import streamlit as st

from .model import FLEX_CSV, FEATURES, PRESETS, Conditions, load_flex, predict, sweep_o2, train
from .prompts import explain_prediction_prompt
from .ui import _call


@st.cache_resource
def _load_models(mtime_ns):
    return train(load_flex())


def _apply_preset():
    preset = PRESETS[st.session_state.predict_preset]
    st.session_state.predict_fuel = preset.fuel
    st.session_state.predict_pressure = float(preset.pressure_psi)
    st.session_state.predict_o2 = float(preset.o2_pct)
    st.session_state.predict_co2 = float(preset.co2_pct)
    st.session_state.predict_he = float(preset.he_pct)
    st.session_state.predict_d0 = float(preset.d0_mm)


def _conditions():
    if "predict_preset" not in st.session_state:
        st.session_state.predict_preset = next(iter(PRESETS))
        _apply_preset()
    st.selectbox("Preset", list(PRESETS), key="predict_preset", on_change=_apply_preset)
    fuel = st.selectbox("Fuel", ["Methanol", "Heptane"], key="predict_fuel")
    pressure = st.number_input("Pressure (psi)", 0.5, 45.0, step=0.1, key="predict_pressure")
    o2 = st.number_input("O₂ (%)", 5.0, 60.0, step=0.5, key="predict_o2")
    co2 = st.number_input("CO₂ (%)", 0.0, 70.0, step=0.5, key="predict_co2")
    he = st.number_input("He (%)", 0.0, 50.0, step=0.5, key="predict_he")
    d0 = st.number_input("Initial droplet diameter (mm)", 1.0, 6.0, step=0.1, key="predict_d0")
    return Conditions(fuel, pressure, o2, co2, he, d0)


def _model_card(models, count):
    metrics = models.metrics
    with st.expander("Model card"):
        st.markdown(f"FLEX PSI-69: {count} ISS tests ([NASA PSI](https://psi.nasa.gov/physci/repo/data/investigations/PSI-69)).")
        st.write("Features: " + ", ".join(FEATURES))
        st.write(f"5-fold CV outcome accuracy: {metrics['accuracy']:.2f}; majority baseline: {metrics['baseline_accuracy']:.2f}.")
        for name in ("extinction_d_mm", "burning_rate", "burn_time_s"):
            st.write(f"{name}: CV R² {metrics['r2'][name]:.2f} (n={metrics['n'][name]}).")
        st.caption("Only methanol and heptane droplets; small sample. Not for real fire-safety decisions.")


def render_predict(ready: bool):
    st.subheader("Predict (own ML model)")
    if not FLEX_CSV.exists():
        st.warning("FLEX table is missing. Run `python scripts/fetch_tables.py` to download it.")
        return
    models = _load_models(FLEX_CSV.stat().st_mtime_ns)
    conditions = _conditions()
    if conditions.o2_pct + conditions.co2_pct + conditions.he_pct > 100:
        st.error("O₂ + CO₂ + He cannot exceed 100%.")
        return
    result = predict(models, conditions)
    columns = st.columns(4)
    probability = result["outcome_proba"][result["outcome"]]
    columns[0].metric("Likely outcome", f"{result['outcome']} ({probability:.0%})")
    columns[1].metric("Burn time", f"{result['burn_time_s']:.1f} s")
    columns[2].metric("Burning rate", f"{result['burning_rate']:.2f} mm")
    columns[3].metric("Extinction diameter", f"{result['extinction_d_mm']:.2f} mm")
    st.bar_chart(pd.Series(result["outcome_proba"], name="Probability"))
    if result["out_of_range"]:
        details = ", ".join(
            f"{item['feature']}={item['value']:.3g} (training {item['min']:.3g}–{item['max']:.3g})"
            for item in result["out_of_range"]
        )
        st.warning(f"extrapolation beyond ISS test range: {details}")
    st.write("Outcome probabilities as O₂ rises (12–40%)")
    sweep = sweep_o2(models, conditions, range(12, 41))
    st.line_chart(sweep.set_index("o2_pct")[list(models.classes)])
    o2_min, o2_max = models.feature_ranges["o2"]
    st.caption(
        f"ISS training data covers O₂ {o2_min * 100:.0f}–{o2_max * 100:.0f}%; "
        "values outside this range are extrapolation."
    )
    _model_card(models, len(load_flex()))
    if st.button("Explain with SI", disabled=not ready):
        explanation = _call("prediction", explain_prediction_prompt(conditions, result, models.metrics), ready)
        if explanation is not None:
            st.markdown(explanation)
    if not ready:
        st.info("Add ANTHROPIC_API_KEY or GEMINI_API_KEY to .env to enable SI explanations.")
