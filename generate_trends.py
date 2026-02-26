#!/usr/bin/env python3
"""
Generate deterministic QE trend output JSON from an Excel file.

Usage:
  python scripts/generate_trends.py --input data/input.xlsx --output output/trends.json

Rules:
- Strict grouping: Event Subcategory (EV) → Event Defect Code (EV)
- Semantic similarity uses ONLY: Title (QE) + Direct cause details (QE)
- Day of Created Date (QE) is read (if present) but NEVER used for clustering
- Trend = >= 3 cohesive events within the same Subcategory+Defect group
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_FIELDS: Dict[str, List[str]] = {
    "Name (QE)": ["name (qe)", "qe", "qe number", "event id", "nummer", "name", "qe-nr", "qe nr"],
    "Title (QE)": ["title (qe)", "titel", "title", "beschreibung", "short description", "kurztext"],
    "Event Subcategory (EV)": ["event subcategory (ev)", "subcategory", "sub category", "unterkategorie", "sub-kategorie"],
    "Event Defect Code (EV)": ["event defect code (ev)", "defect code", "defect", "fehlercode", "code", "defektcode"],
    "Direct cause details (QE)": ["direct cause details (qe)", "direct cause", "ursache", "root cause", "cause details", "direkte ursache"],
    "Day of Created Date (QE)": ["day of created date (qe)", "created date", "erstellungsdatum", "datum", "created day"],
}

GERMAN_STOPWORDS = {
    "und","oder","aber","wenn","dann","weil","da","dass","die","der","das","ein","eine","einer","eines",
    "ist","sind","war","waren","wird","werden","wurde","wurden","mit","ohne","für","von","im","in","am","an",
    "auf","aus","bei","bis","durch","gegen","ins","über","unter","um","zu","zum","zur","nach","vor","hinter",
    "nicht","kein","keine","keinen","keinem","keiner","nur","auch","sehr","mehr","weniger","wie","als",
    "dies","diese","dieser","dieses","hier","dort","sowie","bzw","bsp","z","zb","u","ua",
    "quality","event","qe","ev","code","defect","subcategory","subcat","details","cause","direct","created"
}
EN_STOPWORDS = {
    "the","a","an","and","or","but","if","then","because","as","that","this","these","those",
    "is","are","was","were","be","been","being",
    "with","without","for","from","in","on","at","by","to","into","over","under","about","between",
    "not","no","none","only","also","very","more","less","than","as",
}
STOPWORDS = sorted(GERMAN_STOPWORDS | EN_STOPWORDS)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def map_headers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    cols_norm = {c: _norm(c) for c in df.columns}
    mapping: Dict[str, str] = {}
    for req, aliases in REQUIRED_FIELDS.items():
        aliases_norm = set(_norm(a) for a in aliases + [req])
        found = None
        for col, coln in cols_norm.items():
            if coln in aliases_norm:
                found = col
                break
        if found is None:
            for col, coln in cols_norm.items():
                if any(a in coln for a in aliases_norm):
                    found = col
                    break
        if found is not None:
            mapping[req] = found
    ren = {orig: req for req, orig in mapping.items()}
    return df.rename(columns=ren).copy(), mapping

def clean_text(x: object) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    s = str(x).replace("\n"," ").replace("\r"," ").replace("\t"," ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def make_semantic_text(df: pd.DataFrame) -> pd.Series:
    title = df.get("Title (QE)", "").apply(clean_text) if "Title (QE)" in df.columns else ""
    cause = df.get("Direct cause details (QE)", "").apply(clean_text) if "Direct cause details (QE)" in df.columns else ""
    combo = (title.astype(str) + " — " + cause.astype(str)).str.strip(" —")
    return combo.fillna("").astype(str)

def build_similarity_matrix(texts: List[str]) -> np.ndarray:
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
    n = len(texts)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([0])

    S = build_similarity_matrix(texts)
    D = 1.0 - S
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=float(1.0 - sim_threshold),
        metric="precomputed",
        linkage="average",
    )
    return model.fit_predict(D)

def cluster_quality_gate(texts: List[str], labels: np.ndarray, cohesion_threshold: float = 0.45) -> Dict[int, List[int]]:
    if len(texts) == 0:
        return {}
    S = build_similarity_matrix(texts)
    clusters: Dict[int, List[int]] = {}
    labels_list = labels.tolist()
    for lab in sorted(set(labels_list)):
        idx = [i for i, l in enumerate(labels_list) if l == lab]
        if len(idx) < 3:
            continue
        sub = S[np.ix_(idx, idx)]
        mean_sim = (sub.sum() - len(idx)) / (len(idx) * (len(idx) - 1))
        if mean_sim >= cohesion_threshold:
            clusters[lab] = idx
    return clusters

def top_keywords(texts: List[str], k: int = 5) -> List[str]:
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
    picks: List[str] = []
    for i in order:
        t = terms[i]
        if re.fullmatch(r"\d+", t):
            continue
        if t in {"problem", "fehler", "abweichung"}:
            continue
        picks.append(t)
        if len(picks) >= k:
            break
    return picks

def sentence_trend_title(keywords: List[str]) -> str:
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
    uniq: List[str] = []
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
    must_have = ["Name (QE)", "Title (QE)", "Event Subcategory (EV)", "Event Defect Code (EV)", "Direct cause details (QE)"]
    missing = [c for c in must_have if c not in df.columns]
    if missing:
        return {"meta": {"status": "error", "missing_required_fields": missing}, "trends": [], "no_trend_groups": []}

    for c in must_have + ["Day of Created Date (QE)"]:
        if c in df.columns:
            df[c] = df[c].apply(clean_text)

    df = df[df["Event Subcategory (EV)"].astype(str).str.strip().ne("")]
    df = df[df["Event Defect Code (EV)"].astype(str).str.strip().ne("")]
    df = df.reset_index(drop=True)
    df["__semantic_text"] = make_semantic_text(df)

    trends: List[Dict] = []
    no_trend_groups: List[Dict] = []

    subcats = sorted(df["Event Subcategory (EV)"].unique().tolist(), key=lambda x: str(x))
    for sub in subcats:
        df_sub = df[df["Event Subcategory (EV)"] == sub]
        defects = sorted(df_sub["Event Defect Code (EV)"].unique().tolist(), key=lambda x: str(x))
        for defect in defects:
            group = df_sub[df_sub["Event Defect Code (EV)"] == defect].copy().reset_index(drop=True)

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

            cluster_items = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            for lab, idxs in cluster_items:
                cdf = group.iloc[idxs].copy()
                kws = top_keywords(cdf["__semantic_text"].tolist(), k=5)
                trend_name = sentence_trend_title(kws)
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
                    "Trend Name": trend_name,
                    "Trend Summary": summary,
                    "Number of Events in Trend": int(len(cdf)),
                    "List of QE Numbers (Name (QE))": cdf["Name (QE)"].tolist(),
                    "Aggregated Event Titles": compact_titles(cdf["Title (QE)"].tolist()),
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input .xlsx")
    ap.add_argument("--output", required=True, help="Path to output trends.json")
    args = ap.parse_args()

    inp = Path(args.input)
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_excel(inp, sheet_name=0, engine="openpyxl")
    df_raw = df_raw.dropna(how="all")
    df, mapping = map_headers(df_raw)

    result = analyze(df)
    result["meta"]["header_mapping"] = mapping

    outp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {outp} (status={result['meta'].get('status')}, trends={len(result.get('trends', []))})")


if __name__ == "__main__":
    main()
