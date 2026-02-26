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
