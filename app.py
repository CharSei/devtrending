
import json
from pathlib import Path
import altair as alt
import pandas as pd
import streamlit as st

st.set_page_config(page_title="QE Trend Dashboard", page_icon="📈", layout="wide")
st.title("📈 Quality Event Trend Dashboard")

st.write(
    "Dieses Dashboard identifiziert wiederkehrende Muster (Trends) innerhalb von "
    "**Event Subcategory → Event Defect Code** und zeigt übersichtlich, welche Events dazugehören."
)

DEFAULT_JSON_PATH = Path("output/trends.json")

def load_trends_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def trends_to_tables(payload: dict):
    rows_trends = []
    rows_events = []
    for g in payload.get("groups", []):
        sub = g.get("event_subcategory")
        defect = g.get("event_defect_code")
        gkey = f"{sub} | {defect}"
        # if no trends
        if not g.get("trends"):
            rows_trends.append({
                "Group": gkey,
                "Event Subcategory (EV)": sub,
                "Event Defect Code (EV)": defect,
                "Trend Name": "Kein wiederkehrender Trend identifiziert.",
                "Trend Summary": g.get("no_trend_reason",""),
                "Number of Events in Trend": 0,
                "QE Numbers": "",
            })
            for ev in g.get("events", []):
                rows_events.append({
                    "Group": gkey,
                    "Event Subcategory (EV)": sub,
                    "Event Defect Code (EV)": defect,
                    "Trend Name": "—",
                    "QE Number": ev.get("qe_number"),
                    "Title (QE)": ev.get("title"),
                    "Direct cause details (QE)": ev.get("direct_cause_details"),
                    "Day of Created Date (QE)": ev.get("day_of_created_date"),
                })
            continue

        for t in g["trends"]:
            rows_trends.append({
                "Group": gkey,
                "Event Subcategory (EV)": sub,
                "Event Defect Code (EV)": defect,
                "Trend Name": t.get("trend_name"),
                "Trend Summary": t.get("trend_summary"),
                "Number of Events in Trend": t.get("n_events", 0),
                "QE Numbers": ", ".join(t.get("qe_numbers", [])),
            })
            for ev in t.get("events", []):
                rows_events.append({
                    "Group": gkey,
                    "Event Subcategory (EV)": sub,
                    "Event Defect Code (EV)": defect,
                    "Trend Name": t.get("trend_name"),
                    "QE Number": ev.get("qe_number"),
                    "Title (QE)": ev.get("title"),
                    "Direct cause details (QE)": ev.get("direct_cause_details"),
                    "Day of Created Date (QE)": ev.get("day_of_created_date"),
                })
    df_trends = pd.DataFrame(rows_trends)
    df_events = pd.DataFrame(rows_events)
    return df_trends, df_events

with st.sidebar:
    st.header("Datenquelle")
    mode = st.radio("Quelle auswählen", ["Repo-Output (output/trends.json)", "JSON hochladen"], index=0)
    payload = None
    if mode.startswith("Repo-Output"):
        if DEFAULT_JSON_PATH.exists():
            payload = load_trends_json(DEFAULT_JSON_PATH)
            st.success(f"Geladen: {DEFAULT_JSON_PATH.as_posix()}")
        else:
            st.warning("Keine output/trends.json gefunden. Nutze GitHub Actions oder lade JSON hoch.")
    else:
        up = st.file_uploader("Trend-JSON hochladen", type=["json"])
        if up is not None:
            payload = json.load(up)
            st.success("JSON hochgeladen.")

    st.divider()
    st.header("Filter")
    if payload:
        df_trends, df_events = trends_to_tables(payload)
        subcats = ["(Alle)"] + sorted([x for x in df_trends["Event Subcategory (EV)"].dropna().unique().tolist()])
        defects = ["(Alle)"] + sorted([x for x in df_trends["Event Defect Code (EV)"].dropna().unique().tolist()])
        f_sub = st.selectbox("Event Subcategory (EV)", subcats, index=0)
        f_def = st.selectbox("Event Defect Code (EV)", defects, index=0)
        min_n = st.slider("Min. Events pro Trend", 0, int(max(0, df_trends["Number of Events in Trend"].max() if len(df_trends) else 0)), 3)
        search = st.text_input("Suche in Trendname / Titel / Ursache", "")

if not payload:
    st.info("Bitte eine Datenquelle auswählen (Repo-Output oder JSON Upload).")
    st.stop()

df_trends, df_events = trends_to_tables(payload)

# Apply filters
mask = pd.Series([True]*len(df_trends))
if f_sub != "(Alle)":
    mask &= df_trends["Event Subcategory (EV)"].eq(f_sub)
if f_def != "(Alle)":
    mask &= df_trends["Event Defect Code (EV)"].eq(f_def)
mask &= df_trends["Number of Events in Trend"].ge(min_n)

df_tr_f = df_trends[mask].copy()

if search.strip():
    s = search.strip().lower()
    # match in trends
    m1 = df_tr_f["Trend Name"].fillna("").str.lower().str.contains(s) | df_tr_f["Trend Summary"].fillna("").str.lower().str.contains(s)
    df_tr_f = df_tr_f[m1].copy()

# Events filtered consistent with trends selection + search
df_ev_f = df_events.copy()
if f_sub != "(Alle)":
    df_ev_f = df_ev_f[df_ev_f["Event Subcategory (EV)"].eq(f_sub)]
if f_def != "(Alle)":
    df_ev_f = df_ev_f[df_ev_f["Event Defect Code (EV)"].eq(f_def)]
if search.strip():
    s = search.strip().lower()
    df_ev_f = df_ev_f[
        df_ev_f["Trend Name"].fillna("").str.lower().str.contains(s) |
        df_ev_f["Title (QE)"].fillna("").str.lower().str.contains(s) |
        df_ev_f["Direct cause details (QE)"].fillna("").str.lower().str.contains(s)
    ]

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Gruppen", int(df_events["Group"].nunique()))
c2.metric("Trends (≥ Filter)", int(df_tr_f[df_tr_f["Number of Events in Trend"]>0].shape[0]))
c3.metric("Events (gefiltert)", int(df_ev_f.shape[0]))
c4.metric("Events gesamt", int(df_events.shape[0]))

st.divider()

# Charts row
left, right = st.columns([1.2, 1])

# Top trends chart
df_top = df_tr_f[df_tr_f["Number of Events in Trend"]>0].copy()
df_top = df_top.sort_values("Number of Events in Trend", ascending=False).head(15)

with left:
    st.subheader("Top Trends nach Anzahl Events")
    if len(df_top):
        chart = (
            alt.Chart(df_top)
            .mark_bar()
            .encode(
                y=alt.Y("Trend Name:N", sort="-x", title="Trend"),
                x=alt.X("Number of Events in Trend:Q", title="Anzahl Events"),
                tooltip=["Event Subcategory (EV)", "Event Defect Code (EV)", "Number of Events in Trend", "Trend Summary"]
            )
            .properties(height=420)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Keine Trends für die aktuelle Filterauswahl.")

# Heatmap by group size and trend size
with right:
    st.subheader("Problem-Heatmap nach Gruppe")
    df_grp = df_events.groupby(["Event Subcategory (EV)", "Event Defect Code (EV)"], dropna=False).size().reset_index(name="Events in Group")
    df_grp_tr = df_events[df_events["Trend Name"].ne("—")].groupby(["Event Subcategory (EV)", "Event Defect Code (EV)"]).size().reset_index(name="Events in Trends")
    df_hm = df_grp.merge(df_grp_tr, on=["Event Subcategory (EV)", "Event Defect Code (EV)"], how="left").fillna(0)
    df_hm["Events in Trends"] = df_hm["Events in Trends"].astype(int)
    df_hm["Events in Group"] = df_hm["Events in Group"].astype(int)
    metric = st.radio("Heatmap-Metrik", ["Events in Group", "Events in Trends"], horizontal=True, index=1)
    hm = (
        alt.Chart(df_hm)
        .mark_rect()
        .encode(
            x=alt.X("Event Defect Code (EV):N", title="Event Defect Code (EV)"),
            y=alt.Y("Event Subcategory (EV):N", title="Event Subcategory (EV)"),
            color=alt.Color(f"{metric}:Q", title=metric),
            tooltip=["Event Subcategory (EV)", "Event Defect Code (EV)", "Events in Group", "Events in Trends"]
        )
        .properties(height=420)
    )
    st.altair_chart(hm, use_container_width=True)

st.divider()

st.subheader("Trend-Übersicht")
st.dataframe(
    df_tr_f.sort_values(["Event Subcategory (EV)", "Event Defect Code (EV)", "Number of Events in Trend"], ascending=[True, True, False]),
    use_container_width=True,
    hide_index=True,
)

st.divider()
st.subheader("Trends innerhalb der Gruppen (mit zugehörigen Events)")

# Build hierarchical view
groups_order = (
    df_ev_f[["Event Subcategory (EV)", "Event Defect Code (EV)"]]
    .drop_duplicates()
    .sort_values(["Event Subcategory (EV)", "Event Defect Code (EV)"])
    .values.tolist()
)

for sub, defect in groups_order:
    gkey = f"{sub} | {defect}"
    with st.expander(f"{gkey}", expanded=False):
        # show group summary
        n_group = int(df_events[df_events["Group"].eq(gkey)].shape[0])
        n_trend_events = int(df_events[(df_events["Group"].eq(gkey)) & (df_events["Trend Name"].ne("—"))].shape[0])
        st.caption(f"Events in Group: {n_group} • Events in Trends: {n_trend_events}")

        # trends in this group
        t_in = df_trends[(df_trends["Group"].eq(gkey)) & (df_trends["Number of Events in Trend"]>0)].copy()
        if f_sub != "(Alle)":
            t_in = t_in[t_in["Event Subcategory (EV)"].eq(f_sub)]
        if f_def != "(Alle)":
            t_in = t_in[t_in["Event Defect Code (EV)"].eq(f_def)]
        if search.strip():
            s = search.strip().lower()
            t_in = t_in[t_in["Trend Name"].fillna("").str.lower().str.contains(s) | t_in["Trend Summary"].fillna("").str.lower().str.contains(s)]
        if min_n:
            t_in = t_in[t_in["Number of Events in Trend"].ge(min_n)]

        if t_in.empty:
            st.info("Kein Trend in dieser Gruppe (unter den aktuellen Filtern).")
        else:
            for _, tr in t_in.sort_values("Number of Events in Trend", ascending=False).iterrows():
                st.markdown(f"**{tr['Trend Name']}**")
                if tr.get("Trend Summary"):
                    st.write(tr["Trend Summary"])
                # events table for this trend
                ev = df_ev_f[(df_ev_f["Group"].eq(gkey)) & (df_ev_f["Trend Name"].eq(tr["Trend Name"]))].copy()
                ev = ev[["QE Number", "Title (QE)", "Direct cause details (QE)", "Day of Created Date (QE)"]]
                st.dataframe(ev, use_container_width=True, hide_index=True)
                st.divider()

st.divider()
st.subheader("Export")
colA, colB = st.columns([1, 1])
with colA:
    st.download_button(
        "⬇️ Trend-JSON herunterladen",
        data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="trends.json",
        mime="application/json",
    )
with colB:
    st.caption("Hinweis: Die Analyse wird in der Hybrid-Architektur deterministisch geclustert; "
               "LLM wird optional nur für Trend-Namen/Summaries genutzt (über GitHub Actions).")
