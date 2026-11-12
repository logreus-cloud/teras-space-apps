import pytest

from teras.model import FLEX_CSV, FEATURES, OUTCOME, PRESETS, TARGETS, Conditions, load_flex, predict, sweep_o2, to_features, train


@pytest.fixture(scope="module")
def flex_data():
    if not FLEX_CSV.exists():
        pytest.skip("FLEX CSV is not available")
    return load_flex()


@pytest.fixture(scope="module")
def models(flex_data):
    return train(flex_data)


def test_load_flex(flex_data):
    assert 250 <= len(flex_data) <= 274
    assert flex_data[OUTCOME].notna().all()
    assert flex_data[["o2", "d0_mm"]].notna().all().all()
    assert flex_data[[*FEATURES, *TARGETS]].select_dtypes(include="object").empty
    assert flex_data[[*FEATURES, *TARGETS]].isna().any().any()


def test_train(models):
    assert models.metrics["accuracy"] > models.metrics["baseline_accuracy"]
    assert models.metrics["r2"]["extinction_d_mm"] > 0.5


def test_predict_iss(models):
    conditions = PRESETS["ISS cabin (21% O2, 14.7 psi)"]
    result = predict(models, conditions)
    assert sum(result["outcome_proba"].values()) == pytest.approx(1, abs=1e-6)
    assert result["outcome"] in models.classes
    assert result["out_of_range"] == []


def test_predict_flags_high_oxygen(models):
    conditions = Conditions("Methanol", 14.7, 60, 0, 0, 2.0)
    assert "o2" in {item["feature"] for item in predict(models, conditions)["out_of_range"]}


def test_sweep_o2(models):
    values = [12, 21, 34, 40]
    result = sweep_o2(models, PRESETS["ISS cabin (21% O2, 14.7 psi)"], values)
    assert len(result) == len(values)
    assert result["o2_pct"].tolist() == values
    assert "burn_time_s" in result


def test_to_features():
    values = to_features(Conditions("Heptane", 14.7, 21, 15, 10, 3.0))
    assert values["n2"] == pytest.approx(0.54)
    assert values["pressure_mmhg"] == pytest.approx(14.7 * 51.715)
