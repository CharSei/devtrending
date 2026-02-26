# QE Trend Dashboard (Hybrid)

## Überblick
- Deterministische Trend-Struktur (Subcategory → Defect Code → Cluster aus Title + Direct cause details)
- Optional: LLM erzeugt **Trend-Namen als ganze Sätze** + prägnante Summaries (Deutsch)
- Streamlit Dashboard mit Filtern, Top-Trends, Heatmap und Gruppen-Detailansicht inkl. zugehöriger Events

## Lokal starten
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Hybrid-Workflow (GitHub Actions)
1. Lege deine Excel-Datei als `data/input.xlsx` ins Repo
2. Push auf `main` → Workflow generiert `output/trends.json` und committed zurück

### Optional: LLM für Trend-Namen/Summaries
- Lege ein Repo Secret an: `OPENAI_API_KEY`
- Optional: `OPENAI_MODEL` (z.B. `gpt-4o-mini`), Standard ist im Workflow gesetzt.

Ohne API Key läuft die Pipeline **rein deterministisch** (Fallback-Texte).

## Dateien
- `scripts/generate_trends.py`: Deterministische Cluster + optional LLM-Postprocessing
- `.github/workflows/generate_trends.yml`: Automation
- `output/trends.json`: Dashboard Input
