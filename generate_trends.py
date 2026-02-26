#!/usr/bin/env python3
"""Generate trends.json from latest *.xlsx in repo root (no API).
This script mirrors the deterministic trend engine used in the Streamlit app.
"""

import json
import re
from pathlib import Path
import sys
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack


REQUIRED_FIELDS = [
    "Name (QE)",
    "Title (QE)",
    "Event Subcategory (EV)",
    "Event Defect Code (EV)",
    "Direct cause details (QE)",
    "Day of Created Date (QE)",
]

def _clean_text(x: str) -> str:
    if x is None:
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

def _tokenize(t: str):
    t = _clean_text(t).lower().replace("/", " ")
    out = []
    for w in t.split():
        w = "".join(ch for ch in w if ch.isalnum())
        if len(w) >= 4:
            out.append(w)
    return out

def _top_phrases(texts, top_k=8):
    stop = set([
        "und","oder","der","die","das","mit","auf","in","von","für","ist","eine","ein","bei","wurde","werden","nicht",
        "zu","als","aufgrund","im","am","an","aus","nach","vor","während","the","and","or","of","to","in","on","for",
        "with","is","are","was","were","issue","problem","found","noted"
    ])
    from collections import Counter
    cnt = Counter()
    for t in texts:
        toks = [w for w in _tokenize(t) if w not in stop]
        for i in range(len(toks)-1):
            cnt[f"{toks[i]} {toks[i+1]}"] += 1
        for i in range(len(toks)-2):
            cnt[f"{toks[i]} {toks[i+1]} {toks[i+2]}"] += 1
    phrases = [p for p,_ in cnt.most_common(top_k)]
    if not phrases:
        c2 = Counter()
        for t in texts:
            for w in [w for w in _tokenize(t) if w not in stop]:
                c2[w] += 1
        phrases = [w for w,_ in c2.most_common(top_k)]
    return phrases

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
    v_word = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    v_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=1)
    Xw = v_word.fit_transform(texts)
    Xc = v_char.fit_transform(texts)
    X = hstack([Xw.multiply(0.70), Xc.multiply(0.30)])
    return cosine_similarity(X)

def _connected_components(sim, threshold):
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
            neigh = [v for v in range(n) if (v != u and sim[u, v] >= threshold)]
            for v in neigh:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comps.append(sorted(comp))
    comps.sort(key=lambda c: (-len(c), c))
    return comps

def _cohesion(sim, idxs):
    if len(idxs) < 2:
        return 0.0
    sub = sim[np.ix_(idxs, idxs)]
    tri = sub[np.triu_indices(len(idxs), k=1)]
    return float(tri.mean()) if tri.size else 0.0

def _representatives(sim, idxs, k=3):
    sub = sim[np.ix_(idxs, idxs)]
    scores = sub.mean(axis=1)
    order = np.argsort(-scores)
    reps = [idxs[int(i)] for i in order[:min(k, len(idxs))]]
    return reps

def generate_trends(df: pd.DataFrame, sim_threshold: float = 0.62, cohesion_min: float = 0.58):
    df = _map_headers(df)
    trends = []
    group_stats = []

    grouped = df.groupby(["Event Subcategory (EV)", "Event Defect Code (EV)"], dropna=False, sort=True)

    for (subcat, defect), g in grouped:
        subcat = subcat if subcat else "UNSPECIFIED"
        defect = defect if defect else "UNSPECIFIED"

        titles = g["Title (QE)"].tolist()
        causes = g["Direct cause details (QE)"].tolist()
        ids = g["Name (QE)"].tolist()

        group_stats.append({
            "subcategory": subcat,
            "defect_code": defect,
            "n_events_group": int(len(g)),
        })

        if len(g) < 3:
            continue

        sem = [f"{_clean_text(t)} | {_clean_text(c)}" for t, c in zip(titles, causes)]
        sim = _build_similarity(sem)
        comps = _connected_components(sim, threshold=float(sim_threshold))

        for comp in comps:
            if len(comp) < 3:
                continue
            coh = _cohesion(sim, comp)
            if coh < float(cohesion_min):
                continue

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
                "trend_title": trend_title,
                "trend_summary": summary,
                "n_events": int(len(comp)),
                "similarity": round(coh, 3),
                "qe_numbers": comp_ids,
                "sample_titles": examples,
                "patterns": phrases[:10],
            })

    trends.sort(key=lambda t: (-t["n_events"], -t["similarity"], t["subcategory"], t["defect_code"], t["trend_title"]))

    return {
        "meta": {
            "version": "deviations-trending-mvp-v4",
            "trend_definition": "Connected components on TFIDF similarity graph within Subcategory+Defect (Title+DirectCause).",
            "created_date_note": "Day of Created Date is never used for clustering.",
            "parameters": {
                "sim_threshold_edge": sim_threshold,
                "cohesion_min": cohesion_min,
            }
        },
        "group_rollup": group_stats,
        "trends": trends,
    }

def find_latest_xlsx():
    files = sorted([p for p in Path('.').glob('*.xlsx') if p.is_file()], key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None

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
