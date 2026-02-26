import json
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Quality Event Trend Analyzer", page_icon="📈", layout="wide")
st.title("📈 Quality Event Trend Analyzer")
st.write(
    "Lade eine Excel-Datei mit Quality Events hoch. Die App gruppiert strikt nach "
    "**Event Subcategory (EV)** → **Event Defect Code (EV)** und sucht anschließend "
    "wiederkehrende Muster über semantische Ähnlichkeit in **Title (QE)** + **Direct cause details (QE)**.\n\n"
    "Regel: Ein Trend wird nur gebildet, wenn innerhalb derselben Subcategory+Defect-Kombination "
    "mindestens **3** thematisch klar ähnliche Events gefunden werden."
)

REQUIRED_FIELDS: Dict[str, List[str]] = {
    "Name (QE)": ["name (qe)", "qe", "qe number", "event id", "nummer", "name", "qe-nr", "qe nr"],
    "Title (QE)": ["title (qe)", "titel", "title", "beschreibung", "short description", "kurztext"],
    "Event Subcategory (EV)": ["event subcategory (ev)", "subcategory", "sub category", "unterkategorie", "sub-kategorie"],
    "Event Defect Code (EV)": ["event defect code (ev)", "defect code", "defect", "fehlercode", "code", "defektcode"],
    "Direct cause details (QE)": ["direct cause details (qe)", "direct cause", "ursache", "root cause", "cause details", "direkte ursache"],
    "Day of Created Date (QE)": ["day of created date (qe)", "created date", "erstellungsdatum", "datum", "created day"],
}

GERMAN_STOPWORDS = {
    # small but effective, deterministic list (no external downloads)
    "und","oder","aber","wenn","dann","weil","da","dass","die","der","das","ein","eine","einer","eines",
    "ist","sind","war","waren","wird","werden","wurde","wurden","mit","ohne","für","von","im","in","am","an",
    "auf","aus","bei","bis","durch","gegen","ins","über","unter","um","zu","zum","zur","nach","vor","hinter",
    "nicht","kein","keine","keinen","keinem","keiner","nur","auch","sehr","mehr","weniger","wie","als",
    "dies","diese","dieser","dieses","hier","dort","sowie","bzw","bsp","z","zb","u","ua",
    # common QE boilerplate
    "quality","event","qe","ev","code","defect","subcategory","subcat","details","cause","direct","created"
}
EN_STOPWORDS = {
    "the","a","an","and","or","but","if","then","because","as","that","this","these","those",
    "is","are","was","were","be","been","being",
    "with","without","for","from","in","on","at","by","to","into","over","under","about","between",
    "not","no","none","only","also","very","more","less","than","as",
}
STOPWORDS = sorted(GERMAN_STOPWORDS | EN_STOPWORDS)


# ----------------------------
# Helpers: header mapping + cleaning
# ----------------------------
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def map_headers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Map arbitrary headers to the REQUIRED_FIELDS keys.
    Returns (df_renamed, mapping_required_to_original).
    """
    cols_norm = {c: _norm(c) for c in df.columns}
    mapping: Dict[str, str] = {}

    for req, aliases in REQUIRED_FIELDS.items():
        aliases_norm = set(_norm(a) for a in aliases + [req])
        found = None
        for col, coln in cols_norm.items():
            if coln in aliases_norm:
                found = col
                break
        # fallback: contains match (e.g., "Event Defect Code" inside longer header)
        if found is None:
            for col, coln in cols_norm.items():
                if any(a in coln for a in aliases_norm):
                    found = col
                    break
        if found is not None:
            mapping[req] = found

    # Rename only the mapped columns to required names
    ren = {orig: req for req, orig in mapping.items()}
    df2 = df.rename(columns=ren).copy()
    return df2, mapping

def clean_text(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def make_semantic_text(df: pd.DataFrame) -> pd.Series:
    title = df.get("Title (QE)", "").apply(clean_text) if "Title (QE)" in df.columns else ""
    cause = df.get("Direct cause details (QE)", "").apply(clean_text) if "Direct cause details (QE)" in df.columns else ""
    combo = (title.astype(str) + " — " + cause.astype(str)).str.strip(" —")
    return combo.fillna("").astype(str)

def load_first_visible_sheet(uploaded: BytesIO) -> pd.DataFrame:
    # pandas reads first sheet by default; keep it deterministic
    return pd.read_excel(uploaded, sheet_name=0, engine="openpyxl")


# ----------------------------
# Clustering logic (more accurate + deterministic)
# ----------------------------
def build_similarity_matrix(texts: List[str]) -> np.ndarray:
    """
    Combine word TF-IDF and character n-gram TF-IDF to improve robustness to typos/variants.
    Returns an NxN similarity matrix in [0,1].
    """
    # Word-level
    word_vec = TfidfVectorizer(
        lowercase=True,
        stop_words=STOPWORDS,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
        token_pattern=r"(?u)\b[\w\-]{2,}\b",
    )
    Xw = word_vec.fit_transform(texts)
    Sw = cosine_similarity(Xw)

    # Char-level (handles spelling variants, part numbers fragments)
    char_vec = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_df=0.95,
    )
    Xc = char_vec.fit_transform(texts)
    Sc = cosine_similarity(Xc)

    S = (0.65 * Sw) + (0.35 * Sc)
    np.fill_diagonal(S, 1.0)
    return S

def cluster_texts(texts: List[str], sim_threshold: float = 0.42) -> np.ndarray:
    """
    Agglomerative clustering on (1 - similarity), with a tuned threshold.
    """
    n = len(texts)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0])

    S = build_similarity_matrix(texts)
    D = 1.0 - S
    # Deterministic: linkage=average, precomputed distance
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=float(1.0 - sim_threshold),
        metric="precomputed",
        linkage="average",
    )
    labels = model.fit_predict(D)
    return labels

def cluster_quality_gate(texts: List[str], labels: np.ndarray, cohesion_threshold: float = 0.45) -> Dict[int, List[int]]:
    """
    Keep only clusters that are cohesive enough (mean internal similarity).
    """
    if len(texts) == 0:
        return {}
    S = build_similarity_matrix(texts)
    clusters: Dict[int, List[int]] = {}
    for lab in sorted(set(labels.tolist())):
        idx = [i for i, l in enumerate(labels.tolist()) if l == lab]
        if len(idx) < 3:
            continue
        # mean pairwise similarity (excluding diagonal)
        sub = S[np.ix_(idx, idx)]
        if len(idx) > 1:
            mean_sim = (sub.sum() - len(idx)) / (len(idx) * (len(idx) - 1))
        else:
            mean_sim = 0.0
        if mean_sim >= cohesion_threshold:
            clusters[lab] = idx
    return clusters

def top_keywords(texts: List[str], k: int = 5) -> List[str]:
    """
    Extract top keywords (deterministic) from a set of texts using TF-IDF.
    """
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words=STOPWORDS,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        token_pattern=r"(?u)\b[\w\-]{2,}\b",
    )
    X = vec.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    order = scores.argsort()[::-1]
    picks = []
    for i in order:
        t = terms[i]
        # remove pure numbers / overly generic tokens
        if re.fullmatch(r"\d+", t):
            continue
        if t in {"problem", "fehler", "abweichung"}:
            continue
        picks.append(t)
        if len(picks) >= k:
            break
    return picks

def sentence_trend_title(keywords: List[str]) -> str:
    """
    Return a full German sentence (Trend Name requirement).
    """
    if not keywords:
        return "Mehrere Quality Events weisen ein wiederkehrendes Muster mit ähnlicher Ursache auf."
    if len(keywords) == 1:
        phrase = keywords[0]
    elif len(keywords) == 2:
        phrase = f"{keywords[0]} und {keywords[1]}"
    else:
        phrase = ", ".join(keywords[:-1]) + f" und {keywords[-1]}"
    return f"Mehrere Quality Events weisen wiederkehrende Probleme im Zusammenhang mit {phrase} auf."

def compact_titles(titles: List[str], max_len: int = 320) -> str:
    uniq = []
    seen = set()
    for t in titles:
        tt = clean_text(t)
        if not tt:
            continue
        key = tt.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(tt)
    merged = " | ".join(uniq)
    if len(merged) <= max_len:
        return merged
    return merged[: max_len - 3].rstrip() + "..."

def analyze(df: pd.DataFrame) -> Dict:
    """
    Output schema:
    {
      "meta": {...},
      "trends": [ {trend objects...} ],
      "no_trend_groups": [ {group objects...} ]
    }
    """
    df = df.copy()

    # Minimal required columns check
    must_have = ["Name (QE)", "Title (QE)", "Event Subcategory (EV)", "Event Defect Code (EV)", "Direct cause details (QE)"]
    missing = [c for c in must_have if c not in df.columns]
    if missing:
        return {
            "meta": {"status": "error", "missing_required_fields": missing},
            "trends": [],
            "no_trend_groups": [],
        }

    # Clean key columns
    for c in must_have + ["Day of Created Date (QE)"]:
        if c in df.columns:
            df[c] = df[c].apply(clean_text)

    # Drop empty rows for hierarchy keys
    df = df[df["Event Subcategory (EV)"].astype(str).str.strip().ne("")]
    df = df[df["Event Defect Code (EV)"].astype(str).str.strip().ne("")]
    df = df.reset_index(drop=True)

    df["__semantic_text"] = make_semantic_text(df)

    trends: List[Dict] = []
    no_trend_groups: List[Dict] = []

    # strict order: sort groups deterministically
    subcats = sorted(df["Event Subcategory (EV)"].unique().tolist(), key=lambda x: str(x))
    for sub in subcats:
        df_sub = df[df["Event Subcategory (EV)"] == sub]
        defects = sorted(df_sub["Event Defect Code (EV)"].unique().tolist(), key=lambda x: str(x))
        for defect in defects:
            group = df_sub[df_sub["Event Defect Code (EV)"] == defect].copy()
            group = group.reset_index(drop=True)

            if len(group) < 3:
                no_trend_groups.append({
                    "Event Subcategory (EV)": sub,
                    "Event Defect Code (EV)": defect,
                    "status": "no_recurring_trend",
                    "reason": "fewer_than_3_events_in_group",
                    "number_of_events_in_group": int(len(group)),
                    "qe_numbers": group["Name (QE)"].tolist(),
                })
                continue

            texts = group["__semantic_text"].fillna("").astype(str).tolist()
            labels = cluster_texts(texts, sim_threshold=0.42)
            clusters = cluster_quality_gate(texts, labels, cohesion_threshold=0.45)

            if not clusters:
                no_trend_groups.append({
                    "Event Subcategory (EV)": sub,
                    "Event Defect Code (EV)": defect,
                    "status": "no_recurring_trend",
                    "reason": "no_cluster_meets_similarity_threshold",
                    "number_of_events_in_group": int(len(group)),
                    "qe_numbers": group["Name (QE)"].tolist(),
                })
                continue

            # Create one trend per kept cluster
            # Sort clusters by size desc then label
            cluster_items = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            for cluster_rank, (lab, idxs) in enumerate(cluster_items, start=1):
                cluster_df = group.iloc[idxs].copy()
                kws = top_keywords(cluster_df["__semantic_text"].tolist(), k=5)
                trend_name = sentence_trend_title(kws)

                # Summary: 1–2 sentences explaining similarity
                if kws:
                    summary = (
                        "Die Ereignisse innerhalb dieser Gruppe teilen wiederkehrende Formulierungen und Ursachenmuster, "
                        f"insbesondere rund um {', '.join(kws[:3])}. "
                        "Die Ähnlichkeit wurde ausschließlich aus Title und Direct cause details abgeleitet."
                    )
                else:
                    summary = (
                        "Die Ereignisse innerhalb dieser Gruppe zeigen ein wiederkehrendes sprachliches und ursachenbezogenes Muster. "
                        "Die Ähnlichkeit wurde ausschließlich aus Title und Direct cause details abgeleitet."
                    )

                trends.append({
                    "Event Subcategory (EV)": sub,
                    "Event Defect Code (EV)": defect,
                    "Trend Name": trend_name,  # full sentence
                    "Trend Summary": summary,
                    "Number of Events in Trend": int(len(cluster_df)),
                    "List of QE Numbers (Name (QE))": cluster_df["Name (QE)"].tolist(),
                    "Aggregated Event Titles": compact_titles(cluster_df["Title (QE)"].tolist()),
                })

    return {
        "meta": {
            "status": "ok",
            "n_events_analyzed": int(len(df)),
            "n_trends": int(len(trends)),
            "trend_rule": ">=3 similar events within same Subcategory+Defect (Title+Direct cause details only).",
        },
        "trends": trends,
        "no_trend_groups": no_trend_groups,
    }


# ----------------------------
# UI
# ----------------------------
st.sidebar.header("Datenquelle")

source_mode = st.sidebar.radio(
    "Quelle",
    ["Excel hochladen", "Repo-Output (output/trends.json) anzeigen"],
    index=0,
)

uploaded_file = None
trend_json: Optional[Dict] = None

if source_mode == "Excel hochladen":
    uploaded_file = st.file_uploader("Excel (.xlsx)", type=["xlsx"])
    if uploaded_file is not None:
        try:
            df_raw = load_first_visible_sheet(uploaded_file)
        except Exception as e:
            st.error(f"Excel konnte nicht gelesen werden: {e}")
            st.stop()

        df_raw = df_raw.dropna(how="all")
        df_mapped, mapping = map_headers(df_raw)

        st.sidebar.subheader("Erkannte Spalten")
        if mapping:
            st.sidebar.json(mapping)
        else:
            st.sidebar.warning("Keine passenden Spalten gefunden. Bitte Header prüfen.")

        trend_json = analyze(df_mapped)

elif source_mode == "Repo-Output (output/trends.json) anzeigen":
    try:
        with open("output/trends.json", "r", encoding="utf-8") as f:
            trend_json = json.load(f)
        st.sidebar.success("output/trends.json geladen.")
    except Exception as e:
        st.sidebar.error(f"Konnte output/trends.json nicht laden: {e}")

if trend_json is None:
    st.info("Bitte Excel hochladen oder Repo-Output auswählen.")
    st.stop()

# Error handling
if trend_json.get("meta", {}).get("status") != "ok":
    st.error("Analyse konnte nicht ausgeführt werden.")
    st.json(trend_json.get("meta", {}))
    st.stop()

trends = trend_json.get("trends", [])
no_trend_groups = trend_json.get("no_trend_groups", [])

# ----------------------------
# Overview metrics
# ----------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Analysierte Events", trend_json["meta"].get("n_events_analyzed", 0))
c2.metric("Gefundene Trends", trend_json["meta"].get("n_trends", 0))
c3.metric("Gruppen ohne Trend", len(no_trend_groups))

st.divider()

# ----------------------------
# Charts: where are the biggest problems?
# ----------------------------
st.subheader("Grafische Trend-Übersicht")

trend_df = pd.DataFrame(trends)
if not trend_df.empty:
    # Top trends by count
    top_n = st.slider("Top Trends (nach Anzahl Events)", 5, 30, 15)
    td = trend_df.sort_values("Number of Events in Trend", ascending=False).head(top_n).copy()
    td["Trend Short"] = td["Trend Name"].str.slice(0, 80) + np.where(td["Trend Name"].str.len() > 80, "...", "")

    bar = (
        alt.Chart(td)
        .mark_bar()
        .encode(
            x=alt.X("Number of Events in Trend:Q", title="Anzahl Events im Trend"),
            y=alt.Y("Trend Short:N", sort="-x", title="Trend (gekürzt)"),
            tooltip=[
                alt.Tooltip("Event Subcategory (EV):N"),
                alt.Tooltip("Event Defect Code (EV):N"),
                alt.Tooltip("Number of Events in Trend:Q"),
                alt.Tooltip("Trend Name:N"),
            ],
        )
        .properties(height=min(500, 22 * len(td) + 60))
    )
    st.altair_chart(bar, use_container_width=True)

    # Heatmap: total events per Subcategory+Defect (from input trends + no_trend_groups)
    st.caption("Heatmap zeigt die Gruppengröße je Subcategory+Defect (unabhängig davon, ob ein Trend erkannt wurde).")
    # Reconstruct group sizes deterministically
    group_sizes = {}
    for t in trends:
        k = (t["Event Subcategory (EV)"], t["Event Defect Code (EV)"])
        group_sizes[k] = group_sizes.get(k, 0) + int(t["Number of Events in Trend"])
    for g in no_trend_groups:
        k = (g["Event Subcategory (EV)"], g["Event Defect Code (EV)"])
        # no_trend_groups contains full group size, but may overlap with trends? (it doesn't in our logic)
        group_sizes[k] = max(group_sizes.get(k, 0), int(g.get("number_of_events_in_group", 0)))

    heat_df = pd.DataFrame(
        [{"Event Subcategory (EV)": k[0], "Event Defect Code (EV)": k[1], "Events in Group": v} for k, v in group_sizes.items()]
    )
    heat = (
        alt.Chart(heat_df)
        .mark_rect()
        .encode(
            x=alt.X("Event Defect Code (EV):N", title="Event Defect Code (EV)"),
            y=alt.Y("Event Subcategory (EV):N", title="Event Subcategory (EV)"),
            tooltip=["Event Subcategory (EV)", "Event Defect Code (EV)", "Events in Group"],
            color=alt.Color("Events in Group:Q", title="Events"),
        )
        .properties(height=min(500, 24 * max(1, heat_df["Event Subcategory (EV)"].nunique()) + 60))
    )
    st.altair_chart(heat, use_container_width=True)
else:
    st.info("Keine Trends gefunden – daher keine Trend-Grafiken verfügbar.")

st.divider()

# ----------------------------
# Filters + tables
# ----------------------------
st.subheader("Trend-Details")

if trend_df.empty:
    st.warning("Es wurden keine Trends identifiziert.")
else:
    subcats = ["(alle)"] + sorted(trend_df["Event Subcategory (EV)"].unique().tolist())
    sel_sub = st.selectbox("Filter: Event Subcategory (EV)", subcats, index=0)

    df_view = trend_df.copy()
    if sel_sub != "(alle)":
        df_view = df_view[df_view["Event Subcategory (EV)"] == sel_sub]

    defects = ["(alle)"] + sorted(df_view["Event Defect Code (EV)"].unique().tolist())
    sel_def = st.selectbox("Filter: Event Defect Code (EV)", defects, index=0)
    if sel_def != "(alle)":
        df_view = df_view[df_view["Event Defect Code (EV)"] == sel_def]

    df_show = df_view[
        [
            "Event Subcategory (EV)",
            "Event Defect Code (EV)",
            "Trend Name",
            "Number of Events in Trend",
            "Aggregated Event Titles",
        ]
    ].sort_values(["Number of Events in Trend", "Event Subcategory (EV)", "Event Defect Code (EV)"], ascending=[False, True, True])

    st.dataframe(df_show, use_container_width=True, hide_index=True)

    st.subheader("Einzelne Trends (QE Nummern)")
    for _, row in df_view.sort_values("Number of Events in Trend", ascending=False).iterrows():
        header = f'{row["Event Subcategory (EV)"]} → {row["Event Defect Code (EV)"]} | {row["Number of Events in Trend"]} Events'
        with st.expander(header, expanded=False):
            st.markdown(f"**Trend Name:** {row['Trend Name']}")
            st.markdown(f"**Trend Summary:** {row['Trend Summary']}")
            st.markdown("**QE Numbers:**")
            st.code(", ".join(row["List of QE Numbers (Name (QE))"]), language="text")
            st.markdown("**Aggregated Titles:**")
            st.write(row["Aggregated Event Titles"])

st.divider()

st.subheader("Download / Export")
st.download_button(
    "Trends JSON herunterladen",
    data=json.dumps(trend_json, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name="trends.json",
    mime="application/json",
)
with st.expander("JSON Vorschau"):
    st.json(trend_json)
