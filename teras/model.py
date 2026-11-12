from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

FLEX_CSV = Path(__file__).resolve().parents[1] / "data/raw/PSI-69_Experimental table_FLEX.csv"
FEATURES = ("heptane", "pressure_mmhg", "o2", "n2", "co2", "he", "d0_mm")
TARGETS = {
    "extinction_d_mm": "Visible flame extinction diameter; mm",
    "burning_rate": "Burning rate; mm",
    "burn_time_s": "Burn time; s",
}
OUTCOME = "test_end"


def load_flex(path=FLEX_CSV) -> pd.DataFrame:
    source = pd.read_csv(path, encoding="latin-1")
    clean = pd.DataFrame(index=source.index)
    clean["heptane"] = source["Fuel"].eq("Heptane").astype(int)
    columns = {
        "pressure_mmhg": "Ambient pressure; mmHg",
        "o2": "O initial ambient composition; mole fraction",
        "n2": "N initial ambient composition; mole fraction",
        "co2": "CO initial ambient composition; mole fraction",
        "he": "He initial ambient composition; mole fraction",
        "d0_mm": "Droplet initial diameter; mm",
        **TARGETS,
    }
    for target, source_name in columns.items():
        values = pd.to_numeric(source[source_name], errors="coerce")
        if target in ("n2", "co2", "he"):
            values = values.where(source[source_name].notna(), 0)
        clean[target] = values
    clean["pressure_mmhg"] = clean["pressure_mmhg"].where(clean["pressure_mmhg"] > 0)
    clean[OUTCOME] = source["Test end"].astype("string").str.strip().replace("", pd.NA)
    return clean.dropna(subset=[OUTCOME, "o2", "d0_mm"]).loc[:, [*FEATURES, *TARGETS, OUTCOME]]


@dataclass(frozen=True)
class FlexModels:
    classifier: HistGradientBoostingClassifier
    regressors: dict
    metrics: dict
    feature_ranges: dict
    classes: tuple


@dataclass(frozen=True)
class Conditions:
    fuel: str
    pressure_psi: float
    o2_pct: float
    co2_pct: float
    he_pct: float
    d0_mm: float


PRESETS = {
    "ISS cabin (21% O2, 14.7 psi)": Conditions("Methanol", 14.7, 21, 0, 0, 2.0),
    "Lunar habitat (34% O2, 8.2 psi)": Conditions("Methanol", 8.2, 34, 0, 0, 2.0),
    "Exploration atmosphere (30% O2, 10.2 psi)": Conditions("Methanol", 10.2, 30, 0, 0, 2.0),
    "CO2 suppression (21% O2, 15% CO2)": Conditions("Heptane", 14.7, 21, 15, 0, 3.0),
}


def train(df, random_state=0) -> FlexModels:
    x = df.loc[:, list(FEATURES)]
    y = df[OUTCOME]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    classifier = HistGradientBoostingClassifier(max_depth=3, random_state=random_state)
    metrics = {
        "accuracy": float(cross_val_score(classifier, x, y, cv=folds).mean()),
        "baseline_accuracy": float(y.value_counts(normalize=True).max()),
        "r2": {},
        "n": {},
    }
    classifier.fit(x, y)
    regressors = {}
    for target in TARGETS:
        valid = df[target].notna()
        regressor = HistGradientBoostingRegressor(max_depth=3, random_state=random_state)
        folds = KFold(n_splits=5, shuffle=True, random_state=random_state)
        metrics["r2"][target] = float(cross_val_score(regressor, x.loc[valid], df.loc[valid, target], cv=folds).mean())
        metrics["n"][target] = int(valid.sum())
        regressor.fit(x.loc[valid], df.loc[valid, target])
        regressors[target] = regressor
    ranges = {
        name: (float(x[name].min()), float(x[name].max()))
        for name in FEATURES
    }
    return FlexModels(classifier, regressors, metrics, ranges, tuple(classifier.classes_))


def to_features(conditions: Conditions) -> dict:
    return {
        "heptane": int(conditions.fuel == "Heptane"),
        "pressure_mmhg": conditions.pressure_psi * 51.715,
        "o2": conditions.o2_pct / 100,
        "n2": max(0.0, 1 - (conditions.o2_pct + conditions.co2_pct + conditions.he_pct) / 100),
        "co2": conditions.co2_pct / 100,
        "he": conditions.he_pct / 100,
        "d0_mm": conditions.d0_mm,
    }


def predict(models: FlexModels, conditions: Conditions) -> dict:
    features = to_features(conditions)
    frame = pd.DataFrame([features], columns=FEATURES)
    probabilities = dict(zip(models.classes, map(float, models.classifier.predict_proba(frame)[0])))
    out_of_range = [
        {"feature": name, "value": value, "min": low, "max": high}
        for name, value in features.items()
        for low, high in (models.feature_ranges[name],)
        if value < low or value > high
    ]
    return {
        "outcome_proba": probabilities,
        "outcome": max(probabilities, key=probabilities.get),
        **{name: float(model.predict(frame)[0]) for name, model in models.regressors.items()},
        "out_of_range": out_of_range,
    }


def sweep_o2(models: FlexModels, conditions: Conditions, o2_values) -> pd.DataFrame:
    rows = []
    for value in o2_values:
        if float(value) + conditions.co2_pct + conditions.he_pct > 100:
            continue
        result = predict(models, replace(conditions, o2_pct=float(value)))
        rows.append({"o2_pct": float(value), **result["outcome_proba"], "burn_time_s": result["burn_time_s"]})
    return pd.DataFrame(rows)
