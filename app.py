import json
from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering


# -----------------------------
# Deterministic Trend Engine
# -----------------------------

REQUIRED_FIELDS = [
    "Name (QE)",
    "Title (QE)",
    "Event Subcategory (EV)",
    "Event Defect Code (EV)",
    "Direct cause details (QE)",
    "Day of Created Date (QE)",
]

def _clean_text(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    x = str(x)
    x = x.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    x = " ".join(x.split())
    return x.strip()

def _norm_colname(c: str) -> str:
    return " ".join(str(c).strip().lower().replace("\n", " ").split())

def _map_headers(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    norm = {_norm_colname(c): c for c in cols}

    def pick(candidates):
        for cand in candidates:
            for n, orig in norm.items():
                if cand in n:
                    return orig
        return None

    col_name = pick(["name (qe)", "qe number", "qe#", "qe id", "qe-id", "event id", "qe"])
    col_title = pick(["title (qe)", "title", "short description", "beschreibung", "titel"])
    col_subcat = pick(["event subcategory", "subcategory", "sub category", "unterkategorie"])
    col_defect = pick(["event defect code", "defect code", "defect", "fehlercode", "code"])
    col_cause = pick(["direct cause details", "direct cause", "cause details", "cause", "ursache"])
    col_date = pick(["day of created date", "created date", "creation date", "date created", "created", "datum"])

    rename = {}
    if col_name: rename[col_name] = "Name (QE)"
    if col_title: rename[col_title] = "Title (QE)"
    if col_subcat: rename[col_subcat] = "Event Subcategory (EV)"
    if col_defect: rename[col_defect] = "Event Defect Code (EV)"
    if col_cause: rename[col_cause] = "Direct cause details (QE)"
    if col_date: rename[col_date] = "Day of Created Date (QE)"

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

def _keywords(texts, top_k=4):
    # deterministic simple keyword extraction
    stop = set([
        "und","oder","der","die","das","mit","auf","in","von","für","ist","eine","ein","bei",
        "wurde","werden","nicht","zu","als","aufgrund","im","am","an","aus","nach","vor","während",
        "the","and","or","of","to","in","on","for","with","is","are","was","were"
    ])
    from collections import Counter
    cnt = Counter()
    for t in texts:
        t = _clean_text(t).lower().replace("/", " ")
        for w in t.split():
            w = "".join(ch for ch in w if ch.isalnum())
            if len(w) >= 4 and w not in stop:
                cnt[w] += 1
    return [w for w,_ in cnt.most_common(top_k)]

def _trend_sentence(subcat, defect, titles, causes):
    kw = _keywords(titles, 4)
    kw2 = _keywords(causes, 3)
    merged = []
    for w in kw + kw2:
        if w not in merged:
            merged.append(w)
    if merged:
        return f"Mehrere Quality Events zeigen wiederkehrende Probleme im Zusammenhang mit {', '.join(merged[:4])}."
    return f"Mehrere Quality Events innerhalb von {subcat} / {defect} zeigen ein ähnliches Muster."

def _trend_summary(subcat, defect, n, titles, causes):
    kw = _keywords(titles + causes, 6)
    examples = "; ".join([_clean_text(t)[:90] + ("…" if len(_clean_text(t)) > 90 else "") for t in titles[:3] if _clean_text(t)])
    if not examples:
        examples = "—"
    kw_txt = ", ".join(kw) if kw else "—"
    return f"Die Gruppe ({subcat} → {defect}) umfasst {n} Events. Häufige Stichworte: {kw_txt}. Beispiel-Titel: {examples}."

def _cluster_texts(texts, distance_threshold=0.35):
    # returns labels (deterministic) based on TFIDF cosine distance
    if len(texts) == 1:
        return np.array([0])

    # word + char tfidf (robust to typos)
    v_word = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)

    Xw = v_word.fit_transform(texts)
    Xc = v_char.fit_transform(texts)

    # weighted blend
    X = (0.65 * Xw) + (0.35 * Xc)

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
    df = _map_headers(df)

    trends = []
    group_rollup = []  # for heatmaps

    grouped = df.groupby(["Event Subcategory (EV)", "Event Defect Code (EV)"], dropna=False, sort=True)

    for (subcat, defect), g in grouped:
        subcat = subcat if subcat else "UNSPECIFIED"
        defect = defect if defect else "UNSPECIFIED"

        # build semantic text (ONLY title + direct cause)
        sem = (g["Title (QE)"].fillna("") + " | " + g["Direct cause details (QE)"].fillna("")).map(_clean_text).tolist()
        ids = g["Name (QE)"].tolist()

        # stats for dashboard
        group_rollup.append({
            "subcategory": subcat,
            "defect_code": defect,
            "n_events_group": len(g),
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

        # evaluate clusters
        g2 = g.copy()
        g2["__cluster"] = labels

        any_trend = False
        for cid, cg in g2.groupby("__cluster", sort=True):
            if len(cg) < 3:
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
                continue

            any_trend = True
            titles = cg["Title (QE)"].tolist()
            causes = cg["Direct cause details (QE)"].tolist()
            trend_name = _trend_sentence(subcat, defect, titles, causes)
            summary = _trend_summary(subcat, defect, len(cg), titles, causes)

            trends.append({
                "subcategory": subcat,
                "defect_code": defect,
                "trend_name": trend_name,
                "trend_summary": summary,
                "n_events": int(len(cg)),
                "qe_numbers": cg["Name (QE)"].tolist(),
                "aggregated_titles": " | ".join([t for t in titles if t][:12]),
                "cluster_id": int(cid),
                "is_trend": True,
                "cohesion": round(cohesion, 3),
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

    out = {
        "meta": {
            "version": "demo-prototype-no-api",
            "trend_definition": ">=3 Events innerhalb Subcategory+Defect und kohäsiver Textcluster (Title+DirectCause).",
            "note": "Created Date wird nicht fürs Clustering verwendet.",
        },
        "group_rollup": group_rollup,
        "trends": trends,
    }
    return out


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="QE Trends — Demo", page_icon="📊", layout="wide")
st.title("📊 QE Trend Dashboard — Live Demo (kein API)")
st.caption("Upload Excel für Live-Analyse oder lade vorhandenes trends.json.")

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
    if upj is not None:
        data = json.load(upj)
    elif p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        st.warning("Kein trends.json gefunden. Nutze den Live-Upload oder lege trends.json im Repo-Root ab.")
        st.stop()

# download json
st.download_button(
    "⬇️ trends.json herunterladen",
    data=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
    file_name="trends.json",
    mime="application/json",
)

trends = pd.DataFrame(data.get("trends", []))
roll = pd.DataFrame(data.get("group_rollup", []))

# Normalize for display
trends["subcategory"] = trends["subcategory"].fillna("UNSPECIFIED")
trends["defect_code"] = trends["defect_code"].fillna("UNSPECIFIED")
trends["is_trend"] = trends["is_trend"].fillna(False)

# Filters
st.sidebar.header("Filter")
subcats = ["(alle)"] + sorted(trends["subcategory"].unique().tolist())
defects = ["(alle)"] + sorted(trends["defect_code"].unique().tolist())

sel_sub = st.sidebar.selectbox("Event Subcategory", subcats)
sel_def = st.sidebar.selectbox("Event Defect Code", defects)
min_events = st.sidebar.slider("Min. Anzahl Events (Trend)", 3, int(max(3, trends["n_events"].max())), 3)
search = st.sidebar.text_input("Textsuche (Trendname/Summary/Titel)")

f = trends.copy()
if sel_sub != "(alle)":
    f = f[f["subcategory"] == sel_sub]
if sel_def != "(alle)":
    f = f[f["defect_code"] == sel_def]

# only keep actual trends for trend list view; non-trend shown in group drilldown
f_trends = f[f["is_trend"] == True].copy()
f_trends = f_trends[f_trends["n_events"] >= min_events]

if search.strip():
    s = search.strip().lower()
    def _match(row):
        blob = " ".join([
            str(row.get("trend_name","") or ""),
            str(row.get("trend_summary","") or ""),
            str(row.get("aggregated_titles","") or ""),
        ]).lower()
        return s in blob
    f_trends = f_trends[f_trends.apply(_match, axis=1)]

# KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trends (gefiltert)", int(len(f_trends)))
c2.metric("Events in Trends", int(f_trends["n_events"].sum()) if not f_trends.empty else 0)
c3.metric("Gruppen (gesamt)", int(len(roll)) if not roll.empty else 0)
c4.metric("Max Trendgröße", int(f_trends["n_events"].max()) if not f_trends.empty else 0)

st.divider()

# Charts with drilldown selection
st.subheader("🔥 Wo liegen die größten Probleme?")

sel = alt.selection_point(fields=["subcategory","defect_code","trend_name"], empty=True, name="pick")

bar_df = f_trends.sort_values("n_events", ascending=False).head(30)
if bar_df.empty:
    st.info("Keine Trends für diese Filterkombination gefunden.")
else:
    bar = alt.Chart(bar_df).mark_bar().encode(
        x=alt.X("n_events:Q", title="Anzahl Events im Trend"),
        y=alt.Y("trend_name:N", sort="-x", title="Trend (ganzer Satz)"),
        tooltip=["subcategory","defect_code","n_events","trend_name"]
    ).add_params(sel).transform_filter(sel | alt.datum.trend_name != None).properties(height=420)
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
    ).add_params(heat_sel).properties(height=420)
    st.altair_chart(heat, use_container_width=True)

st.divider()

# Drilldown: Group overview -> trends and events
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
    else:
        for _, row in g_tr.iterrows():
            tname = row["trend_name"]
            n = int(row["n_events"])
            cohesion = row.get("cohesion", None)
            header = f"{tname}  (n={n}" + (f", cohesion={cohesion}" if cohesion is not None else "") + ")"
            with st.expander(header, expanded=True):
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

st.divider()
with st.expander("🔎 Raw JSON Preview", expanded=False):
    st.json(data)
