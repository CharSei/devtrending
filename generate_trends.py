#!/usr/bin/env python3
"""Generate trends.json from latest *.xlsx in repo root (no API).
This script mirrors the deterministic trend engine used in the Streamlit app.
"""

import json
import re
from pathlib import Path
from io import BytesIO

@@ -14,7 +15,7 @@


# -----------------------------
# Deterministic Trend Engine
# Deterministic Trend Engine (v4)
# -----------------------------

REQUIRED_FIELDS = [
@@ -27,7 +28,7 @@
]

def _clean_text(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
    if x is None:
        return ""
    x = str(x)
    x = x.replace("\n", " ").replace("\r", " ").replace("\t", " ")
@@ -65,202 +66,224 @@ def pick(candidates):

    df = df.rename(columns=rename)

    # ensure required exist
    for f in REQUIRED_FIELDS:
        if f not in df.columns:
            df[f] = ""

    # clean values
    for f in REQUIRED_FIELDS:
        df[f] = df[f].apply(_clean_text)

    # drop empty rows (no QE id and no title)
    df = df[~((df["Name (QE)"] == "") & (df["Title (QE)"] == ""))].copy()
    return df

def _keywords_phrases(texts, top_k=6):
    """Deterministische Phrasenextraktion (Bigrams/Trigrams) für verständliche Trendtitel."""
def _tokenize(t: str):
    t = _clean_text(t).lower().replace("/", " ")
    out = []
    for w in t.split():
        w = "".join(ch for ch in w if ch.isalnum())
        if len(w) >= 4:
            out.append(w)
    return out

def _top_phrases(texts, top_k=8):
    # Deterministic phrase extraction (bigrams/trigrams)
    stop = set([
        "und","oder","der","die","das","mit","auf","in","von","für","ist","eine","ein","bei",
        "wurde","werden","nicht","zu","als","aufgrund","im","am","an","aus","nach","vor","während",
        "the","and","or","of","to","in","on","for","with","is","are","was","were","issue","problem"
        "und","oder","der","die","das","mit","auf","in","von","für","ist","eine","ein","bei","wurde","werden","nicht",
        "zu","als","aufgrund","im","am","an","aus","nach","vor","während","the","and","or","of","to","in","on","for",
        "with","is","are","was","were","issue","problem","found","noted"
    ])
    from collections import Counter
    cnt = Counter()
    def tokens(t):
        t = _clean_text(t).lower().replace("/", " ")
        toks = []
        for w in t.split():
            w = "".join(ch for ch in w if ch.isalnum())
            if len(w) >= 4 and w not in stop:
                toks.append(w)
        return toks

    for t in texts:
        toks = tokens(t)
        toks = [w for w in _tokenize(t) if w not in stop]
        for i in range(len(toks)-1):
            cnt[f"{toks[i]} {toks[i+1]}"] += 1
        for i in range(len(toks)-2):
            cnt[f"{toks[i]} {toks[i+1]} {toks[i+2]}"] += 1

    phrases = [p for p,_ in cnt.most_common(top_k)]
    if not phrases:
        c2 = Counter()
        for t in texts:
            for w in tokens(t):
            for w in [w for w in _tokenize(t) if w not in stop]:
                c2[w] += 1
        phrases = [w for w,_ in c2.most_common(top_k)]
    return phrases

def _trend_sentence(subcat, defect, titles, causes):
    """Trendname als klarer deutscher Satz."""
    phrases = _keywords_phrases(titles + causes, top_k=5)
    core = phrases[0] if phrases else "ähnliche Abweichungen"
    return f"In der Gruppe {subcat} → {defect} treten wiederholt Abweichungen im Zusammenhang mit \"{core}\" auf."

def _trend_summary(subcat, defect, n, titles, causes):
    phrases = _keywords_phrases(titles + causes, top_k=6)
    examples = "; ".join([_clean_text(t)[:90] + ("…" if len(_clean_text(t)) > 90 else "") for t in titles[:3] if _clean_text(t)])
    if not examples:
        examples = "—"
    bullets = ", ".join(phrases[:5]) if phrases else "—"
    return f"Die Gruppe ({subcat} → {defect}) umfasst {n} Events mit ähnlicher Beschreibung/Ursache. Häufige Muster: {bullets}. Beispiel-Titel: {examples}."

def _cluster_texts(texts, distance_threshold=0.35):
    # returns labels (deterministic) based on TFIDF cosine distance
    if len(texts) == 1:
        return np.array([0])

    # word + char tfidf (robust to typos)
def _domain_trend_title(titles, causes, phrases):
    """
    Domain-spezifische, deterministische Trendtitel (ohne API).
    Nutzt Regex/Keywords, um verständliche Labels wie 'Steckenbleiben des Personenaufzugs' zu erzeugen.
    """
    text_blob = " ".join([_clean_text(t) for t in titles] + [_clean_text(c) for c in causes]).lower()

    rules = [
        (r"\b(personenaufzug|aufzug|elevator|lift)\b.*\b(stecken|steck|blockier|stillstand|stuck|stopp)\b", "Steckenbleiben des Personenaufzugs"),
        (r"\b(stecken|blockier|stillstand|stuck)\b.*\b(personenaufzug|aufzug|elevator|lift)\b", "Steckenbleiben des Personenaufzugs"),
        (r"\b(dokumentation|nachweis|protokoll|unterlage|bericht|doku)\b.*\b(fehlend|nicht vorhanden|missing|unklar|unvollständig)\b", "Fehlende oder unvollständige Dokumentation"),
        (r"\b(fehlend|nicht vorhanden|missing|unvollständig)\b.*\b(dokumentation|nachweis|protokoll|unterlage|bericht|doku)\b", "Fehlende oder unvollständige Dokumentation"),
    ]

    for pat, label in rules:
        if re.search(pat, text_blob):
            return label

    core = phrases[0] if phrases else "ähnliche Abweichung"
    core = core.replace("_", " ")
    core = core[:60] + ("…" if len(core) > 60 else "")
    return f"Wiederkehrende Abweichung: {core}"

def _domain_trend_summary(subcat, defect, n, phrases, examples):
    """Kurz & verständlich: gemeinsames Muster + Beispiele."""
    patt = ", ".join(phrases[:6]) if phrases else "—"
    ex = "; ".join([e[:110] + ("…" if len(e) > 110 else "") for e in examples]) if examples else "—"
    return (
        f"Gruppe {subcat} → {defect}: {n} ähnliche Events. "
        f"Gemeinsames Muster: {patt}. "
        f"Repräsentative Beispiele: {ex}."
    )

def _build_similarity(texts):
    # robust TFIDF: word + char, concatenated with weights
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from scipy.sparse import hstack

    v_word = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)

    Xw = v_word.fit_transform(texts)
    Xc = v_char.fit_transform(texts)

    # weighted blend
    X = hstack([Xw.multiply(0.65), Xc.multiply(0.35)])

    X = hstack([Xw.multiply(0.70), Xc.multiply(0.30)])
    sim = cosine_similarity(X)
    dist = 1 - sim

    # Agglomerative clustering with precomputed distance
    cl = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold
    )
    labels = cl.fit_predict(dist)
    return labels, sim

def generate_trends(df: pd.DataFrame):
    return sim

def _connected_components(sim, threshold):
    # Deterministic connected components on similarity graph
    n = sim.shape[0]
    visited = [False]*n
    comps = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            # neighbors above threshold (excluding self)
            neigh = [v for v in range(n) if (v != u and sim[u, v] >= threshold)]
            for v in neigh:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comps.append(sorted(comp))
    # sort comps by size desc then lexicographically for determinism
    comps.sort(key=lambda c: (-len(c), c))
    return comps

def _cohesion(sim, idxs):
    if len(idxs) < 2:
        return 0.0
    import numpy as np
    sub = sim[np.ix_(idxs, idxs)]
    tri = sub[np.triu_indices(len(idxs), k=1)]
    return float(tri.mean()) if tri.size else 0.0

def _representatives(sim, idxs, k=3):
    # pick items with highest average similarity to others in cluster
    import numpy as np
    sub = sim[np.ix_(idxs, idxs)]
    scores = sub.mean(axis=1)
    order = np.argsort(-scores)
    reps = [idxs[int(i)] for i in order[:min(k, len(idxs))]]
    return reps

def generate_trends(df: pd.DataFrame, sim_threshold: float = 0.62, cohesion_min: float = 0.58):
    df = _map_headers(df)

    trends = []
    group_rollup = []  # for heatmaps
    group_stats = []

    grouped = df.groupby(["Event Subcategory (EV)", "Event Defect Code (EV)"], dropna=False, sort=True)

    for (subcat, defect), g in grouped:
        subcat = subcat if subcat else "UNSPECIFIED"
        defect = defect if defect else "UNSPECIFIED"

        # build semantic text (ONLY title + direct cause)
        sem = (g["Title (QE)"].fillna("") + " | " + g["Direct cause details (QE)"].fillna("")).map(_clean_text).tolist()
        titles = g["Title (QE)"].tolist()
        causes = g["Direct cause details (QE)"].tolist()
        ids = g["Name (QE)"].tolist()

        # stats for dashboard
        group_rollup.append({
        group_stats.append({
            "subcategory": subcat,
            "defect_code": defect,
            "n_events_group": len(g),
            "n_events_group": int(len(g)),
        })

        # if group too small -> explicit no-trend
        if len(g) < 3:
            trends.append({
                "subcategory": subcat,
                "defect_code": defect,
                "trend_name": None,
                "trend_summary": None,
                "n_events": len(g),
                "qe_numbers": ids,
                "aggregated_titles": " | ".join([t for t in g['Title (QE)'].tolist() if t][:8]),
                "cluster_id": None,
                "is_trend": False,
            })
            continue

        labels, sim = _cluster_texts(sem, distance_threshold=0.35)
        # semantic text (ONLY title + cause)
        sem = [f"{_clean_text(t)} | {_clean_text(c)}" for t, c in zip(titles, causes)]
        sim = _build_similarity(sem)
        comps = _connected_components(sim, threshold=float(sim_threshold))

        # evaluate clusters
        g2 = g.copy()
        g2["__cluster"] = labels

        any_trend = False
        for cid, cg in g2.groupby("__cluster", sort=True):
            if len(cg) < 3:
        for comp in comps:
            if len(comp) < 3:
                continue

            # cohesion gate: average pairwise similarity within cluster
            idx = cg.index.to_list()
            # sim is in order of g rows, map index->position
            pos = {i:p for p,i in enumerate(g.index.to_list())}
            pidx = [pos[i] for i in idx]
            sub_sim = sim[np.ix_(pidx, pidx)]
            # upper triangle average (excluding diagonal)
            if len(pidx) > 1:
                tri = sub_sim[np.triu_indices(len(pidx), k=1)]
                cohesion = float(np.mean(tri)) if tri.size else 0.0
            else:
                cohesion = 0.0

            if cohesion < 0.55:
            coh = _cohesion(sim, comp)
            if coh < float(cohesion_min):
                continue

            any_trend = True
            titles = cg["Title (QE)"].tolist()
            causes = cg["Direct cause details (QE)"].tolist()
            trend_name = _trend_sentence(subcat, defect, titles, causes)
            summary = _trend_summary(subcat, defect, len(cg), titles, causes)
            comp_titles = [titles[i] for i in comp]
            comp_causes = [causes[i] for i in comp]
            comp_ids = [ids[i] for i in comp]

            phrases = _top_phrases(comp_titles + comp_causes, top_k=8)
            core = phrases[0] if phrases else "ähnliche Abweichung"
            trend_title = _domain_trend_title(comp_titles, comp_causes, phrases)

            reps = _representatives(sim, comp, k=3)
            examples = [ _clean_text(titles[i]) for i in reps if _clean_text(titles[i]) ]
            examples_txt = "; ".join([e[:110] + ("…" if len(e) > 110 else "") for e in examples]) if examples else "—"
            patterns_txt = ", ".join(phrases[:6]) if phrases else "—"

            summary = (
                f"In {subcat} → {defect} treten {len(comp)} ähnliche Events auf. "
                f"Häufige Muster: {patterns_txt}. "
                f"Beispiele: {examples_txt}."
            )

            trends.append({
                "subcategory": subcat,
                "defect_code": defect,
                "trend_name": trend_name,
                "trend_title": trend_title,
                "trend_summary": summary,
                "n_events": int(len(cg)),
                "qe_numbers": cg["Name (QE)"].tolist(),
                "aggregated_titles": " | ".join([t for t in titles if t][:12]),
                "cluster_id": int(cid),
                "is_trend": True,
                "cohesion": round(cohesion, 3),
                "n_events": int(len(comp)),
                "similarity": round(coh, 3),
                "qe_numbers": comp_ids,
                "sample_titles": examples,
                "patterns": phrases[:10],
            })

        if not any_trend:
            trends.append({
                "subcategory": subcat,
                "defect_code": defect,
                "trend_name": None,
                "trend_summary": None,
                "n_events": len(g),
                "qe_numbers": ids,
                "aggregated_titles": " | ".join([t for t in g['Title (QE)'].tolist() if t][:8]),
                "cluster_id": None,
                "is_trend": False,
            })
    # Sort trends for deterministic prioritization
    trends.sort(key=lambda t: (-t["n_events"], -t["similarity"], t["subcategory"], t["defect_code"], t["trend_title"]))

    out = {
    return {
        "meta": {
            "version": "demo-prototype-no-api",
            "trend_definition": ">=3 Events innerhalb Subcategory+Defect und kohäsiver Textcluster (Title+DirectCause).",
            "note": "Created Date wird nicht fürs Clustering verwendet.",
            "version": "deviations-trending-mvp-v4",
            "trend_definition": "Connected components on TFIDF similarity graph within Subcategory+Defect (Title+DirectCause).",
            "created_date_note": "Day of Created Date is never used for clustering.",
            "parameters": {
                "sim_threshold_edge": sim_threshold,
                "cohesion_min": cohesion_min,
            }
        },
        "group_rollup": group_rollup,
        "group_rollup": group_stats,
        "trends": trends,
    }
    return out


# -----------------------------
@@ -269,39 +292,30 @@ def generate_trends(df: pd.DataFrame):

st.set_page_config(page_title="Deviations Trending MVP", page_icon="📊", layout="wide")
st.title("📊 Deviations Trending MVP")
st.caption("Upload Excel für Live-Analyse oder lade vorhandenes trends.json.")
st.caption("Live-Demo: Excel hochladen → Trends sofort als sortierte Liste/Tabelle. Keine API notwendig.")

# -----------------------------
# Layout tweaks (13" friendly)
# -----------------------------
st.markdown(
    """
<style>
section[data-testid="stSidebar"] { width: 340px !important; }
section[data-testid="stSidebar"] > div { padding-top: 1rem; }
div[data-testid="stMarkdownContainer"] p { white-space: normal !important; }
section[data-testid="stSidebar"] { width: 360px !important; }
div[data-testid="stMarkdownContainer"] { overflow-wrap: anywhere; }
div[data-testid="stMarkdownContainer"] p { white-space: normal !important; }
code { white-space: pre-wrap !important; }
.block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
.block-container { padding-top: 1.1rem; padding-bottom: 1.1rem; }
</style>
    """,
    unsafe_allow_html=True,
)


mode = st.radio("Modus", ["Live-Analyse (Excel Upload)", "Repository-Modus (trends.json)"], horizontal=True)

data = None
df_events = None

if mode == "Live-Analyse (Excel Upload)":
    up = st.file_uploader("Excel (.xlsx) hochladen", type=["xlsx"])
    if up is None:
        st.info("Bitte eine Excel-Datei hochladen, um die Analyse live zu starten.")
        st.stop()
    # read first visible sheet
    df_in = pd.read_excel(up, sheet_name=0)
    data = generate_trends(df_in)
else:
    p = Path("trends.json")
    upj = st.file_uploader("Optional: trends.json hochladen", type=["json"])
@@ -310,9 +324,19 @@ def generate_trends(df: pd.DataFrame):
    elif p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        st.warning("Kein trends.json gefunden. Nutze den Live-Upload oder lege trends.json im Repo-Root ab.")
        st.warning("Kein trends.json gefunden. Nutze Live-Upload oder lege trends.json im Repo-Root ab.")
        st.stop()

st.sidebar.header("Analyse & Filter")

sim_edge = st.sidebar.slider("Similarity-Kante (Graph) — Edge Threshold", 0.40, 0.85, 0.62, 0.01,
                             help="Ab wann zwei Events als ähnlich verbunden werden.")
cohesion_min = st.sidebar.slider("Trend Similarity — Mindestkohäsion", 0.40, 0.90, 0.58, 0.01,
                                 help="Wie homogen ein Trend-Cluster im Mittel sein muss.")

if mode == "Live-Analyse (Excel Upload)":
    data = generate_trends(df_in, sim_threshold=sim_edge, cohesion_min=cohesion_min)

# download json
st.download_button(
    "⬇️ trends.json herunterladen",
@@ -322,159 +346,106 @@ def generate_trends(df: pd.DataFrame):
)

trends = pd.DataFrame(data.get("trends", []))
roll = pd.DataFrame(data.get("group_rollup", []))

# Normalize for display
trends["subcategory"] = trends["subcategory"].fillna("UNSPECIFIED")
trends["defect_code"] = trends["defect_code"].fillna("UNSPECIFIED")
trends["is_trend"] = trends["is_trend"].fillna(False)
trends["cohesion"] = pd.to_numeric(trends.get("cohesion"), errors="coerce")
if trends.empty:
    st.warning("Keine Trends gefunden. Tipp: Similarity-Kante senken oder Kohäsion-Minimum senken.")
    st.stop()

# Filters
st.sidebar.header("Filter")
subcats = ["(alle)"] + sorted(trends["subcategory"].unique().tolist())
defects = ["(alle)"] + sorted(trends["defect_code"].unique().tolist())

sel_sub = st.sidebar.selectbox("Event Subcategory", subcats)
sel_def = st.sidebar.selectbox("Event Defect Code", defects)
min_cohesion = st.sidebar.slider("Similarity (Kohäsion) — Mindestwert", 0.40, 0.90, 0.55, 0.01)
search = st.sidebar.text_input("Textsuche (Trendname/Summary/Titel)")
min_events = st.sidebar.slider("Min. Events pro Trend", 3, int(max(3, trends["n_events"].max())), 3)
min_sim = st.sidebar.slider("Min. Similarity (Trend)", 0.40, 0.95, 0.58, 0.01)
search = st.sidebar.text_input("Suche (Titel/Summary/Muster)")

f = trends.copy()
if sel_sub != "(alle)":
    f = f[f["subcategory"] == sel_sub]
if sel_def != "(alle)":
    f = f[f["defect_code"] == sel_def]

# only keep actual trends for trend list view; non-trend shown in group drilldown
f_trends = f[f["is_trend"] == True].copy()
f_trends = f_trends[f_trends["cohesion"].fillna(0) >= float(min_cohesion)]
f = f[(f["n_events"] >= int(min_events)) & (f["similarity"] >= float(min_sim))]

if search.strip():
    s = search.strip().lower()
    def _match(row):
    def match(row):
        blob = " ".join([
            str(row.get("trend_name","") or ""),
            str(row.get("trend_title","") or ""),
            str(row.get("trend_summary","") or ""),
            str(row.get("aggregated_titles","") or ""),
            " ".join(row.get("patterns") or [])
        ]).lower()
        return s in blob
    f_trends = f_trends[f_trends.apply(_match, axis=1)]
    f = f[f.apply(match, axis=1)]

# KPI row
# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trends (gefiltert)", int(len(f_trends)))
c2.metric("Events in Trends", int(f_trends["n_events"].sum()) if not f_trends.empty else 0)
c3.metric("Gruppen (gesamt)", int(len(roll)) if not roll.empty else 0)
c4.metric("Ø Similarity", round(float(f_trends["cohesion"].mean()),3) if (not f_trends.empty and "cohesion" in f_trends) else 0)

st.divider()

# Charts with drilldown selection
st.subheader("🔥 Wo liegen die größten Probleme?")

sel = alt.selection_point(fields=["subcategory","defect_code","trend_name"], empty=True, name="pick")

bar_df = f_trends.sort_values("n_events", ascending=False).head(30)
bar_df = bar_df.copy()
bar_df["trend_label"] = bar_df["trend_name"].fillna("").map(lambda s: (s[:70] + "…") if len(s) > 70 else s)
chart_h = max(260, 22 * len(bar_df))
if bar_df.empty:
    st.info("Keine Trends für diese Filterkombination gefunden.")
else:
    bar = alt.Chart(bar_df).mark_bar().encode(
        x=alt.X("n_events:Q", title="Anzahl Events im Trend"),
        y=alt.Y("trend_label:N", sort="-x", title="Trend"),
        tooltip=["subcategory","defect_code","n_events","cohesion","trend_name"]
    ).add_params(sel).properties(height=chart_h)
    st.altair_chart(bar, use_container_width=True)

# Heatmap: total events per group (from rollup)
st.subheader("📌 Heatmap: Event-Last pro Gruppe (Subcategory × Defect)")
if roll.empty:
    st.info("Keine Gruppen-Rollup Daten vorhanden.")
else:
    heat_sel = alt.selection_point(fields=["subcategory","defect_code"], empty=True, name="heat_pick")
    heat = alt.Chart(roll).mark_rect().encode(
        x=alt.X("defect_code:N", title="Defect Code"),
        y=alt.Y("subcategory:N", title="Subcategory"),
        color=alt.Color("n_events_group:Q", title="Events in Gruppe"),
        tooltip=["subcategory","defect_code","n_events_group"]
    ).add_params(heat_sel).properties(height=chart_h)
    st.altair_chart(heat, use_container_width=True)

st.divider()

# Drilldown: Group overview -> trends and events
st.subheader("📋 Trend-Liste (gefiltert)")
if f_trends.empty:
    st.info("Keine Trends für die aktuelle Filterkombination.")
else:
    tbl = f_trends[["subcategory","defect_code","n_events","cohesion","trend_name","trend_summary"]].copy()
    tbl = tbl.sort_values(["n_events","cohesion"], ascending=[False, False])
    st.dataframe(
        tbl,
        use_container_width=True,
        hide_index=True,
        column_config={
            "subcategory": st.column_config.TextColumn("Subcategory"),
            "defect_code": st.column_config.TextColumn("Defect Code"),
            "n_events": st.column_config.NumberColumn("Events", format="%d"),
            "cohesion": st.column_config.NumberColumn("Similarity", format="%.3f"),
            "trend_name": st.column_config.TextColumn("Trend (Satz)"),
            "trend_summary": st.column_config.TextColumn("Zusammenfassung"),
        },
        height=min(520, 36 + 28 * min(len(tbl), 15)),
    )
c1.metric("Trends (gefiltert)", int(len(f)))
c2.metric("Events in Trends", int(f["n_events"].sum()) if not f.empty else 0)
c3.metric("Größter Trend", int(f["n_events"].max()) if not f.empty else 0)
c4.metric("Ø Similarity", round(float(f["similarity"].mean()), 3) if not f.empty else 0)

st.subheader("📋 Trend-Übersicht (priorisiert)")

# Create a nice table view
tbl = f.copy()
tbl = tbl.sort_values(["n_events","similarity","subcategory","defect_code"], ascending=[False, False, True, True])
tbl.insert(0, "rank", range(1, len(tbl)+1))

st.dataframe(
    tbl[["rank","subcategory","defect_code","n_events","similarity","trend_title","trend_summary"]],
    use_container_width=True,
    hide_index=True,
    column_config={
        "rank": st.column_config.NumberColumn("#", format="%d"),
        "subcategory": st.column_config.TextColumn("Subcategory"),
        "defect_code": st.column_config.TextColumn("Defect Code"),
        "n_events": st.column_config.NumberColumn("Events", format="%d"),
        "similarity": st.column_config.NumberColumn("Similarity", format="%.3f"),
        "trend_title": st.column_config.TextColumn("Trend"),
        "trend_summary": st.column_config.TextColumn("Beschreibung"),
    },
    height=min(680, 44 + 28 * min(len(tbl), 18)),
)

st.subheader("🧩 Trends innerhalb der Gruppen + zugehörige Events")

# Determine active selection from charts is not directly available as value in Streamlit,
# so we provide explicit selection widgets for a clean demo flow.
colA, colB = st.columns([2,2])
with colA:
    drill_sub = st.selectbox("Drilldown Subcategory", ["(wähle)"] + sorted(trends["subcategory"].unique().tolist()))
with colB:
    drill_def = st.selectbox("Drilldown Defect Code", ["(wähle)"] + sorted(trends["defect_code"].unique().tolist()))

if drill_sub != "(wähle)" and drill_def != "(wähle)":
    g_all = trends[(trends["subcategory"] == drill_sub) & (trends["defect_code"] == drill_def)].copy()

    # split into trend clusters and non-trend bucket
    g_tr = g_all[g_all["is_trend"] == True].sort_values(["n_events"], ascending=False)
    g_nt = g_all[g_all["is_trend"] == False]

    st.markdown(f"**Gruppe:** `{drill_sub} → {drill_def}`")
    st.write(f"Trends in Gruppe: **{len(g_tr)}**")

    if g_tr.empty:
        st.info("Für diese Gruppe wurde kein wiederkehrender Trend identifiziert (oder Cluster < Schwellenwert).")
        # still show raw events if present in non-trend entry
        if not g_nt.empty:
            qe = g_nt.iloc[0].get("qe_numbers", [])
            st.write(f"Events in Gruppe: **{len(qe)}**")
            st.code("\n".join(qe[:200]))
st.subheader("🔎 Trend-Details")
# Provide a selector for trends (from filtered list)
options = ["(wähle Trend)"] + [
    f"[{int(r.rank)}] {r.trend_title}  —  {r.subcategory} → {r.defect_code}  (n={int(r.n_events)}, sim={float(r.similarity):.3f})"
    for r in tbl.itertuples(index=False)
]
choice = st.selectbox("Trend auswählen", options, index=1 if len(options) > 1 else 0)

if choice != "(wähle Trend)":
    rank = int(choice.split("]")[0].strip("[ "))
    row = tbl[tbl["rank"] == rank].iloc[0].to_dict()

    st.markdown(f"### {row['trend_title']}")
    st.write(row["trend_summary"])

    colx, coly = st.columns([1,2])
    with colx:
        st.metric("Events", int(row["n_events"]))
        st.metric("Similarity", float(row["similarity"]))
        st.write(f"**Gruppe:** `{row['subcategory']} → {row['defect_code']}`")
    with coly:
        st.write("**Häufige Muster (Phrasen):**")
        pats = row.get("patterns") or []
        if pats:
            st.write(", ".join(pats[:10]))
        else:
            st.write("—")

    st.write("**QE Numbers:**")
    st.code("\n".join(row.get("qe_numbers") or []))

    st.write("**Beispiel-Titel:**")
    samples = row.get("sample_titles") or []
    if samples:
        for s in samples:
            st.write(f"- {s}")
    else:
        for _, row in g_tr.iterrows():
            tname = row["trend_name"]
            n = int(row["n_events"])
            cohesion = row.get("cohesion", None)
            header = f"{tname}  (n={n}" + (f", cohesion={cohesion}" if cohesion is not None else "") + ")"
            with st.expander(header if len(header)<=120 else header[:120]+"…", expanded=True):
                st.write(row.get("trend_summary") or "")
                qe = row.get("qe_numbers", [])
                st.write(f"**QE Numbers ({len(qe)}):**")
                st.code("\n".join(qe))

                # show event details table if we can reconstruct from aggregated titles only in JSON
                # In live mode we can compute detailed df; in repo mode trends.json may not carry per-row fields.
                st.caption("Hinweis: Für eine vollständige Event-Tabelle (Title/Cause/Date) bleibt der Live-Upload-Modus am aussagekräftigsten.")
                st.write("**Aggregierte Titel (Auszug):**")
                st.write(row.get("aggregated_titles") or "—")

else:
    st.info("Wähle Subcategory und Defect Code für den Drilldown.")
        st.write("—")

st.divider()
with st.expander("🔎 Raw JSON Preview", expanded=False):
with st.expander("Raw JSON Preview", expanded=False):
    st.json(data)
