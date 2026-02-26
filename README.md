# Quality Event Trend Analyzer (Streamlit)

Streamlit-App zur Trend-Erkennung in Quality Event Daten.

## Features
- Upload von Excel (.xlsx)
- Flexible Header-Erkennung & Mapping auf:
  - Name (QE)
  - Title (QE)
  - Event Subcategory (EV)
  - Event Defect Code (EV)
  - Direct cause details (QE)
  - Day of Created Date (QE) *(wird nur angezeigt/übernommen, aber nie fürs Clustering genutzt)*
- Strikte Hierarchie: **Subcategory → Defect Code**
- Trend-Definition: **mind. 3** thematisch ähnliche Events innerhalb derselben Gruppe
- Export als deterministisches JSON (GitHub/Streamlit-Dashboard-ready)

## Lokal starten
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)
1. Repo auf GitHub pushen
2. In Streamlit Cloud als App auswählen
3. `app.py` als Entry Point
## Automation: trends.json automatisch in GitHub generieren

Dieses Repo enthält eine GitHub Actions Pipeline, die bei Änderungen an `data/input.xlsx`
automatisch `output/trends.json` erzeugt **und zurück ins Repo committed**.

### Dateien
- Workflow: `.github/workflows/generate_trends.yml`
- Script: `scripts/generate_trends.py`
- Input (im Repo): `data/input.xlsx`
- Output (auto-generiert): `output/trends.json`

### Nutzung
1. Lege deine Excel-Datei als **`data/input.xlsx`** ins Repo (Commit & Push).
2. GitHub Actions läuft automatisch und schreibt/updated `output/trends.json`.
3. In Streamlit kannst du `output/trends.json` direkt laden/anzeigen oder weiterhin Upload nutzen.

### Manuell starten (Workflow Dispatch)
GitHub → **Actions** → *Generate QE Trend Output* → **Run workflow**  
Optional kannst du dort `distance_threshold` setzen (Default: `0.35`).
