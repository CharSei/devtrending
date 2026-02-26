#!/usr/bin/env python3
"""Generate trends.json from an Excel file located in repo root (no API required).
- Picks the most recently modified *.xlsx in the repo root.
- Writes trends.json in the repo root.
"""
import json
from pathlib import Path
import sys
import pandas as pd

# Reuse the same logic as in app.py by importing from scripts module would be cleaner,
# but for simplicity in GitHub Actions we keep it self-contained.
# (This file mirrors the deterministic algorithm used in the Streamlit app.)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

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

    for f in REQUIRED_FIELDS:
        if f not in df.columns:
            df[f] = ""

    for f in REQUIRED_FIELDS:
        df[f] = df[f].apply(_clean_text)

    df = df[~((df["Name (QE)"] == "") & (df["Title (QE)"] == ""))].copy()
    return df

def _keywords(texts, top_k=4):
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
    if len(texts) == 1:
        return np.array([0]), np.ones((1,1))

    v_word = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)

    Xw = v_word.fit_transform(texts)
    Xc = v_char.fit_transform(texts)
    X = (0.65 * Xw) + (0.35 * Xc)

    sim = cosine_similarity(X)
    dist = 1 - sim

    cl = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold
    )
    labels = cl.fit_predict(dist)
    return labels, sim

def find_latest_xlsx():
    files = sorted([p for p in Path(".").glob("*.xlsx") if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

def generate_trends(df: pd.DataFrame):
    df = _map_headers(df)

    trends = []
    group_rollup = []

    grouped = df.groupby(["Event Subcategory (EV)", "Event Defect Code (EV)"], dropna=False, sort=True)

    for (subcat, defect), g in grouped:
        subcat = subcat if subcat else "UNSPECIFIED"
        defect = defect if defect else "UNSPECIFIED"

        sem = (g["Title (QE)"].fillna("") + " | " + g["Direct cause details (QE)"].fillna("")).map(_clean_text).tolist()
        ids = g["Name (QE)"].tolist()

        group_rollup.append({
            "subcategory": subcat,
            "defect_code": defect,
            "n_events_group": int(len(g)),
        })

        if len(g) < 3:
            trends.append({
                "subcategory": subcat,
                "defect_code": defect,
                "trend_name": None,
                "trend_summary": None,
                "n_events": int(len(g)),
                "qe_numbers": ids,
                "aggregated_titles": " | ".join([t for t in g['Title (QE)'].tolist() if t][:8]),
                "cluster_id": None,
                "is_trend": False,
            })
            continue

        labels, sim = _cluster_texts(sem, distance_threshold=0.35)

        g2 = g.copy()
        g2["__cluster"] = labels

        any_trend = False
        pos = {i:p for p,i in enumerate(g.index.to_list())}

        for cid, cg in g2.groupby("__cluster", sort=True):
            if len(cg) < 3:
                continue

            idx = cg.index.to_list()
            pidx = [pos[i] for i in idx]
            sub_sim = sim[np.ix_(pidx, pidx)]
            tri = sub_sim[np.triu_indices(len(pidx), k=1)]
            cohesion = float(np.mean(tri)) if tri.size else 0.0
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
                "n_events": int(len(g)),
                "qe_numbers": ids,
                "aggregated_titles": " | ".join([t for t in g['Title (QE)'].tolist() if t][:8]),
                "cluster_id": None,
                "is_trend": False,
            })

    return {
        "meta": {
            "version": "demo-prototype-no-api",
            "trend_definition": ">=3 Events innerhalb Subcategory+Defect und kohäsiver Textcluster (Title+DirectCause).",
            "note": "Created Date wird nicht fürs Clustering verwendet.",
        },
        "group_rollup": group_rollup,
        "trends": trends,
    }

def main():
    xlsx = find_latest_xlsx()
    if not xlsx:
        print("Keine .xlsx im Repo-Root gefunden.", file=sys.stderr)
        sys.exit(1)
    df = pd.read_excel(xlsx, sheet_name=0)
    out = generate_trends(df)
    Path("trends.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: trends.json geschrieben (input={xlsx})")

if __name__ == "__main__":
    main()
