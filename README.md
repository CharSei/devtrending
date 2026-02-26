# Deviations Trending MVP

## Ziel
Schnelle, verständliche Trend-Übersicht als priorisierte Liste/Tabelle – ohne API.

## Live Demo
- Streamlit App starten
- Excel hochladen
- Trends erscheinen als sortierte Tabelle (Events ↓, Similarity ↓)

## Wie ein Trend erkannt wird (deterministisch)
1) Gruppierung: Event Subcategory (EV) → Event Defect Code (EV)  
2) Semantik: nur Title (QE) + Direct cause details (QE)  
3) Similarity-Graph (TF‑IDF Word+Char) → Connected Components → Trend-Cluster (>=3)  
4) Cohesion (mittlere interne Similarity) als Qualitätsfilter

Hinweis: Day of Created Date (QE) wird nie fürs Clustering verwendet.

## Repo Automation (optional)
Lege eine `.xlsx` im Repo-Root ab und pushe. GitHub Actions erzeugt `trends.json`.


## Domain-Regeln (Trendtitel)
Diese MVP-Version nutzt deterministische Regex/Keyword-Regeln für verständliche Trendtitel (z.B. Steckenbleiben des Personenaufzugs, fehlende Dokumentation). Ergänzbar ohne API.
