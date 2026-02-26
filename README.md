# Deviations Trending MVP

## Live Demo
- Starte die App und lade eine `.xlsx` direkt in der UI hoch → Analyse + Dashboard sofort.
- Kein API Key erforderlich (deterministische Trend-Sätze & Cluster).

## Repo / Automation (optional)
- Lege eine `.xlsx` Datei **direkt im Repo-Root** ab (Name egal).
- Push → GitHub Actions erzeugt `trends.json` im Root und committed die Datei.

## Start lokal
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Hinweise
- Analyse-Reihenfolge: Subcategory → Defect Code → Clustering (nur Title + Direct cause details).
- Day of Created Date wird **nie** fürs Clustering verwendet (nur angezeigt/mitgeführt).
