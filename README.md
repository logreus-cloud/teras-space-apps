# Flame in Freefall — SI fire-safety insights from NASA PSI combustion data

Team Teras's Space Apps 2026 dashboard explores combustion investigations and uses Claude or Gemini to summarize, rank, compare, and interpret their relevance to crewed spacecraft fire safety.

## Screenshot

_Dashboard screenshot to be added by the team._

## Data

The included 24 investigations come from [NASA Physical Sciences Informatics (PSI)](https://psi.nasa.gov/). The dashboard reads the local JSON file; the optional fetch script refreshes it from PSI's public API.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
pytest
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

Add `ANTHROPIC_API_KEY` for Claude or `GEMINI_API_KEY` for Gemini to `.env` to enable SI features. Set `LLM_PROVIDER=claude` or `LLM_PROVIDER=gemini` to choose explicitly; with both keys present, Claude is selected by default. Explore and data comparison work without a key.

## Own ML model (Predict tab)

The Predict tab trains a scikit-learn model on NASA FLEX PSI-69 ISS droplet combustion measurements.
Run `python scripts/fetch_tables.py` to download the experimental tables and retrain after the CSV changes.
The Model card shows five-fold cross-validation metrics against the majority-class baseline.
Predictions cover only methanol and heptane droplets from a small sample and must not guide real safety decisions.

## Project structure

- `app.py` — Streamlit entry point and sidebar filters.
- `teras/data.py` — PSI loading and normalization.
- `teras/retrieval.py` — local TF-IDF search.
- `teras/prompts.py`, `teras/llm.py`, `teras/cache.py` — SI prompts, Claude and Gemini access, and response cache.
- `teras/ui.py` — dashboard tabs.
- `scripts/fetch_psi.py` — optional PSI data refresh.
- `tests/` — offline data and search tests.

## Use of SI (AI) tools

The product uses Anthropic Claude or Google Gemini to generate evidence-grounded summaries, scenario rankings, comparisons, answers, and extracted attributes. Responses are cached locally. The team should verify generated claims against the cited PSI investigations.
Predict uses the team's own scikit-learn model trained on NASA measurements.

Claude Opus 5.5 costs $4/$20 per 1M tokens; with the response cache a full demo run stays well under $5. For an even cheaper run set `CLAUDE_MODEL=claude-haiku-5-5`.

Development AI assistants used by the team: _team to complete before submission_.

## Team

Teras has three members:

- Member 1 — _name and role_.
- Member 2 — _name and role_.
- Member 3 — _name and role_.

## License

MIT; see [LICENSE](LICENSE).
