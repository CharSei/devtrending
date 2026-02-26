import json
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Quality Event Trend Analyzer", page_icon="📈", layout="wide")
st.title("📈 Quality Event Trend Analyzer")
st.write(
    "Lade eine Excel-Datei mit Quality Events hoch. Die App gruppiert strikt nach "
    "**Event Subcategory (EV)** → **Event Defect Code (EV)** und sucht anschließend "
    "wiederkehrende Muster über semantische Ähnlichkeit in **Title (QE)** + **Direct cause details (QE)**. "
    "**Day of Created Date (QE)** wird niemals fürs Clustering verwendet."
)

REQUIRED_FIELDS = {
    "Name (QE)": ["name (qe)", "qe", "qe number", "event id", "nummer", "name"],
    "Title (QE)": ["title (qe)", "titel", "title", "beschreibung", "short description"],
    "Event Subcategory (EV)": ["event subcategory (ev)", "subcategory", "sub category", "unterkategorie"],
    "Event Defect Code (EV)": ["event defect code (ev)", "defect code", "defect", "fehlercode", "code"],
    "Direct cause details (QE)": ["direct cause details (qe)", "direct cause", "ursache", "cause details", "root cause"],
    "Day of Created Date (QE)": ["day of created date (qe)", "created date", "datum", "date created"],
}

def _clean_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _normalize_header(h: str) -> str:
    return re.sub(r"\s+", " ", str(h).strip().lower())

def _map_headers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Flexible header mapping. Keeps only the required analytical fields.
    Never fabricates missing data: if a required field can't be mapped, we stop with an error.
    """
    norm_cols = {_normalize_header(c): c for c in df.columns}
    mapping = {}

    for req, candidates in REQUIRED_FIELDS.items():
        found = None
        # exact normalized match
        if _normalize_header(req) in norm_cols:
            found = norm_cols[_normalize_header(req)]
        else:
            # keyword contains match
            for cand in candidates:
                for nc, orig in norm_cols.items():
                    if cand in nc:
                        found = orig
                        break
                if found:
                    break
        if not found:
            raise ValueError(f"Pflichtfeld konnte nicht gefunden werden: {req}")
        mapping[req] = found

    out = df[[mapping[k] for k in mapping.keys()]].copy()
    out.columns = list(mapping.keys())
    return out, mapping

def _cluster_group(texts: List[str], distance_threshold: float, min_size: int = 3):
    """
    Agglomerative clustering over cosine distance on TF-IDF features.
    Returns labels and model artifacts for keyword extraction.
    """
    if len(texts) < min_size:
        return None

    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.9)
    X = vec.fit_transform(texts)

    sim = cosine_similarity(X)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)

    cl = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=float(distance_threshold),
        compute_full_tree=True,
    )
    labels = cl.fit_predict(dist)

    counts = pd.Series(labels).value_counts()
    keep = set(counts[counts >= min_size].index.tolist())

    return {"labels": labels, "keep": keep, "vec": vec, "X": X}

def _top_terms(vec: TfidfVectorizer, X, idxs: np.ndarray, topn: int = 6) -> List[str]:
    sub = X[idxs].mean(axis=0)
    arr = np.asarray(sub).ravel()
    feats = np.array(vec.get_feature_names_out())

    terms = []
    for i in arr.argsort()[::-1]:
        t = feats[i]
        if len(t) <= 2:
            continue
        if re.fullmatch(r"\d+", t):
            continue
        terms.append(t)
        if len(terms) >= topn:
            break
    return terms

def analyze(df_raw: pd.DataFrame, distance_threshold: float = 0.35, min_size: int = 3) -> Dict:
    df, mapping = _map_headers(df_raw)

    for c in df.columns:
        df[c] = df[c].apply(_clean_text)

    # similarity input ONLY (never use created date)
    df["__text"] = df["Title (QE)"] + " " + df["Direct cause details (QE)"]

    groups_out = []
    for (subcat, defect), g in df.groupby(["Event Subcategory (EV)", "Event Defect Code (EV)"], dropna=False):
        g = g.reset_index(drop=True)

        group_obj = {
            "Event Subcategory (EV)": subcat,
            "Event Defect Code (EV)": defect,
            "Total Events": int(len(g)),
            "Trends": [],
        }

        clustered = _cluster_group(g["__text"].tolist(), distance_threshold=distance_threshold, min_size=min_size)

        if clustered is None or not clustered["keep"]:
            group_obj["No Recurring Trend Identified"] = True
        else:
            labels = clustered["labels"]
            keep = clustered["keep"]
            vec = clustered["vec"]
            X = clustered["X"]

            counts = pd.Series(labels).value_counts()

            for cl_id in sorted(list(keep)):
                size = int(counts[cl_id])
                idxs = np.where(labels == cl_id)[0]

                qes = sorted(g.loc[idxs, "Name (QE)"].tolist())
                titles = g.loc[idxs, "Title (QE)"].tolist()
                terms = _top_terms(vec, X, idxs, topn=6)

                trend_name = " / ".join(terms[:3]) if terms else "Ähnliches Muster"
                summary = (
                    f"{size} Events im selben Subcategory+Defect-Code zeigen ein wiederkehrendes Muster "
                    f"(Schlüsselbegriffe: {', '.join(terms)})."
                )

                group_obj["Trends"].append(
                    {
                        "Trend Name": trend_name,
                        "Trend Summary": summary,
                        "Number of Events in Trend": size,
                        "List of QE Numbers (Name (QE))": qes,
                        "Aggregated Event Titles": " | ".join(sorted(set([t.strip() for t in titles]))[:30]),
                    }
                )

            group_obj["No Recurring Trend Identified"] = len(group_obj["Trends"]) == 0

        groups_out.append(group_obj)

    groups_out = sorted(groups_out, key=lambda x: (x["Event Subcategory (EV)"], x["Event Defect Code (EV)"]))

    return {
        "schema_version": "1.0",
        "header_mapping": mapping,
        "parameters": {"distance_threshold": float(distance_threshold), "min_size": int(min_size)},
        "groups": groups_out,
    }


# ----------------------------
# UI
# ----------------------------
with st.sidebar:
    st.header("Eingaben")
    uploaded = st.file_uploader("Excel (.xlsx)", type=["xlsx"])
    distance_threshold = st.slider(
        "Ähnlichkeits-Schwelle (cosine distance)",
        min_value=0.10,
        max_value=0.80,
        value=0.35,
        step=0.01,
        help="Kleiner = strenger (nur sehr ähnliche Texte clustern). Größer = lockerer.",
    )
    st.caption("Trend-Definition ist fest: mind. 3 Events pro Trend.")
    show_only_trends = st.toggle("Nur Gruppen mit Trends anzeigen", value=True)

@st.cache_data(show_spinner=False)
def _load_excel(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes), sheet_name=0)

@st.cache_data(show_spinner=False)
def _run_analysis(df: pd.DataFrame, distance_threshold: float) -> Dict:
    return analyze(df, distance_threshold=distance_threshold, min_size=3)

if not uploaded:
    st.info("Bitte eine Excel-Datei hochladen, um Trends zu berechnen.")
    st.stop()

df_raw = _load_excel(uploaded.getvalue())

try:
    output = _run_analysis(df_raw, distance_threshold)
except Exception as e:
    st.error(f"Analyse fehlgeschlagen: {e}")
    st.stop()

groups = output["groups"]
trend_groups = [g for g in groups if not g.get("No Recurring Trend Identified", True)]

# Header mapping and summary
col1, col2, col3 = st.columns(3)
col1.metric("Gruppen (Subcategory + Defect)", len(groups))
col2.metric("Gruppen mit Trends", len(trend_groups))
col3.metric("Trend-Cluster gesamt", sum(len(g["Trends"]) for g in groups))

with st.expander("Header-Mapping anzeigen"):
    st.json(output["header_mapping"])

# Filters
subcats = sorted(set(g["Event Subcategory (EV)"] for g in groups))
sel_subcat = st.selectbox("Event Subcategory (EV)", ["(alle)"] + subcats, index=0)

defects = sorted(set(g["Event Defect Code (EV)"] for g in groups if sel_subcat in ("(alle)", g["Event Subcategory (EV)"])))
sel_defect = st.selectbox("Event Defect Code (EV)", ["(alle)"] + defects, index=0)

filtered = groups
if show_only_trends:
    filtered = [g for g in filtered if not g.get("No Recurring Trend Identified", True)]
if sel_subcat != "(alle)":
    filtered = [g for g in filtered if g["Event Subcategory (EV)"] == sel_subcat]
if sel_defect != "(alle)":
    filtered = [g for g in filtered if g["Event Defect Code (EV)"] == sel_defect]

# Flatten trends for table
rows = []
for g in filtered:
    if g.get("No Recurring Trend Identified", True):
        continue
    for t in g["Trends"]:
        rows.append({
            "Event Subcategory (EV)": g["Event Subcategory (EV)"],
            "Event Defect Code (EV)": g["Event Defect Code (EV)"],
            "Trend Name": t["Trend Name"],
            "Number of Events": t["Number of Events in Trend"],
            "QE Numbers": ", ".join(t["List of QE Numbers (Name (QE))"]),
            "Titles (aggregated)": t["Aggregated Event Titles"],
            "Summary": t["Trend Summary"],
        })

st.subheader("Trend-Übersicht")
if rows:
    df_trends = pd.DataFrame(rows).sort_values(
        by=["Event Subcategory (EV)", "Event Defect Code (EV)", "Number of Events"],
        ascending=[True, True, False],
    )
    st.dataframe(df_trends, use_container_width=True, hide_index=True)
else:
    st.warning("Keine Trends für die aktuelle Filterauswahl gefunden.")

st.subheader("Gruppen-Detailansicht")
for g in filtered:
    title = f"{g['Event Subcategory (EV)']}  →  {g['Event Defect Code (EV)']}  (n={g['Total Events']})"
    with st.expander(title, expanded=False):
        if g.get("No Recurring Trend Identified", True):
            st.info("Kein wiederkehrender Trend identifiziert (mind. 3 thematisch ähnliche Events erforderlich).")
        else:
            for i, t in enumerate(g["Trends"], start=1):
                st.markdown(f"**Trend {i}: {t['Trend Name']}**  \n{t['Trend Summary']}")
                st.write("**QE Numbers:**", ", ".join(t["List of QE Numbers (Name (QE))"]))
                st.write("**Aggregierte Titles:**", t["Aggregated Event Titles"])

# Downloadable JSON output (GitHub / Dashboard ready)
st.subheader("Export")
json_bytes = json.dumps(output, ensure_ascii=False, indent=2).encode("utf-8")
st.download_button(
    "⬇️ Ergebnis als JSON herunterladen",
    data=json_bytes,
    file_name="qe_trends_output.json",
    mime="application/json",
)

with st.expander("JSON Preview"):
    st.json(output)
