from .data import full_text

SYSTEM_EXPERT = (
    "You are an expert in NASA spacecraft fire safety. "
    "Use only the supplied PSI investigation material. "
    "Cite accessions such as [PSI-69] for factual claims. "
    "Say clearly when evidence is missing and never invent numbers. "
    "Rocket propellant topics are outside scope."
)


def summary_prompt(inv) -> str:
    return (
        f"Investigation {inv.accession}:\n{full_text(inv)}\n\n"
        "Write a structured Markdown summary of at most 250 words. Cover the purpose; "
        "fuel and sample geometry; O2, pressure, flow, and gravity conditions; "
        "measurements and sensors; key findings; implications for fire prevention, "
        "detection, and suppression; and Earth benefits. Identify missing details."
    )


def attributes_prompt(inv) -> str:
    return (
        f"Extract attributes from {inv.accession}:\n{full_text(inv)}\n\n"
        'Return only JSON with keys: fuel_phase (solid|liquid|gas|mixed|unknown), '
        'fuels (array), sample_geometry, oxygen_conditions, flow_conditions (strings), '
        'gravity (microgravity|partial|normal|mixed|unknown), measurements (array), '
        'fire_safety_area (array of prevention|detection|suppression), and '
        'key_findings (array, at most 3). Use "unknown" or [] for missing evidence.'
    )


def rank_prompt(scenario: str, items) -> str:
    lines = []
    for inv in items:
        lines.append(
            f"{inv.accession} | {inv.acronym} | {inv.title} | "
            f"Objectives: {inv.objectives[:600]} | Keywords: {', '.join(inv.keywords)}"
        )
    return (
        f"Scenario: {scenario}\nInvestigations:\n" + "\n".join(lines)
        + '\n\nReturn a JSON array for every investigation: '
        '[{"accession": "PSI-69", "score": 0, "reason": "brief evidence-based reason"}]. '
        "Score relevance from 0 to 100. Use only the supplied evidence."
    )


def compare_prompt(a, b) -> str:
    return (
        f"{a.accession}:\n{full_text(a)}\n\n{b.accession}:\n{full_text(b)}\n\n"
        "Compare fuels, conditions, hardware, measurements, and findings in a "
        "Markdown table, then give three fire-safety insights. Identify missing evidence."
    )


def qa_prompt(question, contexts) -> str:
    evidence = "\n\n".join(
        f"{inv.accession}:\n{full_text(inv)[:3000]}" for inv in contexts
    )
    return (
        f"Question: {question}\n\nPSI evidence:\n{evidence}\n\n"
        "Answer using only this evidence and cite accessions like [PSI-69]."
    )


def explain_prediction_prompt(conditions, prediction, metrics) -> str:
    return (
        f"Conditions: fuel={conditions.fuel}, pressure={conditions.pressure_psi} psi, "
        f"O2={conditions.o2_pct}%, CO2={conditions.co2_pct}%, He={conditions.he_pct}%, "
        f"initial droplet={conditions.d0_mm} mm.\n"
        f"ML prediction: {prediction}\nCV metrics: {metrics}\n\n"
        "In at most 200 words of Markdown, explain the prediction to a spacecraft "
        "fire-safety engineer. Compare it with the ISS cabin preset (21% O2, "
        "14.7 psi), and clearly state any extrapolation beyond the FLEX test range. "
        "Distinguish model estimates from measured results. Cite [PSI-69] for the "
        "FLEX dataset; mention [PSI-68] or [PSI-39] only if relevant, without "
        "inventing findings. State that the model is not for real safety decisions."
    )
