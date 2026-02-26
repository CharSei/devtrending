
"""
Hybrid Trend Generator

1) Deterministic clustering per group (Event Subcategory → Event Defect Code) using ONLY:
   - Title (QE)
   - Direct cause details (QE)
   Day of Created Date is recorded but NEVER used for similarity/clustering.

2) Optional LLM post-processing:
   - Converts trend labels into German full sentences
   - Produces 1–2 sentence summaries
   Enabled when OPENAI_API_KEY is set (GitHub Secret).
   Falls back to deterministic sentence templates if not.

Output is deterministic in structure and stable for dashboard ingestion.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Header mapping (flexible) ---
REQUIRED_FIELDS = {
    "qe_number": ["name (qe)", "qe", "qe number", "qe no", "qe#", "name", "nummer", "id"],
    "title": ["title (qe)", "title", "titel", "beschreibung", "short description", "summary"],
    "event_subcategory": ["event subcategory (ev)", "subcategory", "sub category", "event subcategory", "unterkategorie"],
    "event_defect_code": ["event defect code (ev)", "defect code", "defect", "fehlercode", "code"],
    "direct_cause_details": ["direct cause details (qe)", "direct cause", "cause details", "ursache", "root cause", "cause"],
    "day_of_created_date": ["day of created date (qe)", "created date", "date", "datum", "created", "day"],
}

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())

def map_headers(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: _norm(c) for c in df.columns}
    inv = {v: k for k, v in cols.items()}
    mapping = {}
    for target, aliases in REQUIRED_FIELDS.items():
        found = None
        for a in aliases:
            if a in inv:
                found = inv[a]
                break
        if found is None:
            # fuzzy: contains
            for normed, orig in inv.items():
                if any(a in normed for a in aliases):
                    found = orig
                    break
        if found is not None:
            mapping[found] = target
    df2 = df.rename(columns=mapping).copy()
    return df2

def first_visible_sheet(path: Path) -> pd.DataFrame:
    # pandas/openpyxl loads first sheet by default; that's usually the first visible worksheet.
    return pd.read_excel(path, sheet_name=0)

def clean_text(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_text(title: str, cause: str) -> str:
    # Only semantic fields
    t = clean_text(title)
    c = clean_text(cause)
    if t and c:
        return f"{t}. Ursache: {c}"
    return t or c or ""

@dataclass
class TrendCluster:
    cluster_id: int
    indices: List[int]
    cohesion: float
    keywords: List[str]

def extract_keywords(texts: List[str], top_k: int = 6) -> List[str]:
    vec = TfidfVectorizer(
        stop_words=None,
        ngram_range=(1,2),
        max_features=5000,
        lowercase=True,
    )
    X = vec.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    top = terms[np.argsort(-scores)[:top_k]]
    # cleanup: remove very short tokens
    out = [t for t in top.tolist() if len(t) >= 3]
    return out[:top_k]

def cluster_texts(texts: List[str], distance_threshold: float = 0.55, min_cluster_size: int = 3, cohesion_threshold: float = 0.22) -> Tuple[List[TrendCluster], np.ndarray]:
    """
    Robust similarity: combine word TF-IDF and char ngram TF-IDF.
    Deterministic clustering via AgglomerativeClustering with distance_threshold.
    """
    # If too few texts, no trends
    if len(texts) < min_cluster_size:
        return [], np.array([])

    word_vec = TfidfVectorizer(ngram_range=(1,2), max_features=8000, lowercase=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), max_features=12000)

    Xw = word_vec.fit_transform(texts)
    Xc = char_vec.fit_transform(texts)
    # cosine similarities
    Sw = cosine_similarity(Xw)
    Sc = cosine_similarity(Xc)
    S = 0.65 * Sw + 0.35 * Sc
    # convert to distance for clustering
    D = 1.0 - S
    np.fill_diagonal(D, 0.0)

    # Agglomerative expects condensed? It can take precomputed.
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
    )
    labels = clustering.fit_predict(D)

    clusters: List[TrendCluster] = []
    for cid in sorted(set(labels.tolist())):
        idx = np.where(labels == cid)[0].tolist()
        if len(idx) < min_cluster_size:
            continue
        # cohesion: mean pairwise similarity within cluster (excluding diagonal)
        subS = S[np.ix_(idx, idx)]
        if len(idx) > 1:
            cohesion = float((subS.sum() - len(idx)) / (len(idx)*(len(idx)-1)))
        else:
            cohesion = 0.0
        if cohesion < cohesion_threshold:
            continue
        kw = extract_keywords([texts[i] for i in idx], top_k=6)
        clusters.append(TrendCluster(cluster_id=cid, indices=idx, cohesion=cohesion, keywords=kw))

    return clusters, labels

# --- Optional LLM post-processing ---
def llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())

def llm_generate_trend_sentence_and_summary(subcategory: str, defect: str, keywords: List[str], examples: List[Dict]) -> Tuple[str, str]:
    """
    Returns (trend_name_sentence, trend_summary_1_2_sentences).
    If LLM unavailable, returns deterministic template.
    """
    if not llm_available():
        # deterministic fallback (German, whole sentence)
        kw = ", ".join(keywords[:4]) if keywords else "ähnlichen Ursachen"
        name = f"Mehrere Quality Events zeigen wiederkehrende Probleme im Zusammenhang mit {kw}."
        summary = f"Innerhalb der Gruppe „{subcategory} / {defect}“ treten mehrere Events mit ähnlicher Formulierung und Ursache auf (Schwerpunkt: {kw})."
        return name, summary

    try:
        from openai import OpenAI
        client = OpenAI()
        model = os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini"

        # Provide compact, non-sensitive context: keywords + a few titles/causes (already in dataset)
        ex_lines = []
        for e in examples[:8]:
            t = clean_text(e.get("title",""))[:180]
            c = clean_text(e.get("direct_cause_details",""))[:220]
            ex_lines.append(f"- Titel: {t} | Ursache: {c}")
        prompt = f"""
Du formulierst Trend-Titel und Trend-Zusammenfassung für Qualitätsereignisse.

Vorgaben:
- Sprache: Deutsch
- Trend-Titel: genau 1 ganzer Satz, endet mit Punkt.
- Trend-Zusammenfassung: 1–2 Sätze, neutral, beschreibt den gemeinsamen Mechanismus/Ursachenmuster.
- Keine Personennamen erfinden, keine Daten hinzudichten.
- Nutze Subcategory/Defect als Kontext, aber schreibe verständlich.

Kontext:
Event Subcategory (EV): {subcategory}
Event Defect Code (EV): {defect}
Keywords (automatisch): {", ".join(keywords)}

Beispiel-Events:
{chr(10).join(ex_lines)}
""".strip()

        resp = client.responses.create(
            model=model,
            input=[{"role":"user","content":prompt}],
            temperature=0.2,
        )
        text = resp.output_text.strip()
        # Simple parse: first line title, rest summary
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if len(lines) == 1:
            # split by sentence boundary
            parts = re.split(r"(?<=[.!?])\s+", lines[0], maxsplit=1)
            if len(parts) == 2:
                return parts[0].strip(), parts[1].strip()
            return lines[0].strip(), ""
        title = lines[0]
        summary = " ".join(lines[1:])[:600]
        if not title.endswith("."):
            title += "."
        return title, summary
    except Exception:
        # fallback on any error
        kw = ", ".join(keywords[:4]) if keywords else "ähnlichen Ursachen"
        name = f"Mehrere Quality Events zeigen wiederkehrende Probleme im Zusammenhang mit {kw}."
        summary = f"Innerhalb der Gruppe „{subcategory} / {defect}“ treten mehrere Events mit ähnlicher Formulierung und Ursache auf (Schwerpunkt: {kw})."
        return name, summary

def make_payload(df: pd.DataFrame) -> Dict:
    # ensure required columns exist (don't fabricate; just empty if missing)
    for col in ["qe_number","title","event_subcategory","event_defect_code","direct_cause_details","day_of_created_date"]:
        if col not in df.columns:
            df[col] = ""

    # Clean basic artifacts
    for c in ["qe_number","title","event_subcategory","event_defect_code","direct_cause_details","day_of_created_date"]:
        df[c] = df[c].apply(clean_text)

    groups_out = []
    # strict hierarchy
    for subcat, df_sub in df.groupby("event_subcategory", dropna=False):
        for defect, df_g in df_sub.groupby("event_defect_code", dropna=False):
            records = df_g.to_dict(orient="records")

            texts = [build_text(r.get("title",""), r.get("direct_cause_details","")) for r in records]
            clusters, _labels = cluster_texts(texts)

            if not clusters:
                groups_out.append({
                    "event_subcategory": subcat,
                    "event_defect_code": defect,
                    "no_trend_identified": True,
                    "no_trend_reason": "Entweder weniger als drei thematisch kohärente Events oder die Ähnlichkeit war nicht ausreichend.",
                    "trends": [],
                    "events": [
                        {
                            "qe_number": r.get("qe_number",""),
                            "title": r.get("title",""),
                            "direct_cause_details": r.get("direct_cause_details",""),
                            "day_of_created_date": r.get("day_of_created_date",""),
                        } for r in records
                    ],
                })
                continue

            trends = []
            for k, cl in enumerate(sorted(clusters, key=lambda x: (-len(x.indices), -x.cohesion))):
                evs = [records[i] for i in cl.indices]
                qe_nums = [clean_text(e.get("qe_number","")) for e in evs if clean_text(e.get("qe_number",""))]
                # LLM post-processing (optional)
                trend_name, trend_summary = llm_generate_trend_sentence_and_summary(
                    subcategory=subcat,
                    defect=defect,
                    keywords=cl.keywords,
                    examples=[{
                        "title": e.get("title",""),
                        "direct_cause_details": e.get("direct_cause_details",""),
                    } for e in evs],
                )
                trends.append({
                    "trend_name": trend_name,
                    "trend_summary": trend_summary,
                    "n_events": len(evs),
                    "qe_numbers": qe_nums,
                    "aggregated_event_titles": "; ".join([clean_text(e.get("title","")) for e in evs if clean_text(e.get("title",""))])[:1200],
                    "cohesion": round(cl.cohesion, 4),
                    "keywords": cl.keywords,
                    "events": [
                        {
                            "qe_number": e.get("qe_number",""),
                            "title": e.get("title",""),
                            "direct_cause_details": e.get("direct_cause_details",""),
                            "day_of_created_date": e.get("day_of_created_date",""),
                        } for e in evs
                    ]
                })

            groups_out.append({
                "event_subcategory": subcat,
                "event_defect_code": defect,
                "no_trend_identified": False,
                "trends": trends,
                # optional: include all events for full transparency
                "events": [
                    {
                        "qe_number": r.get("qe_number",""),
                        "title": r.get("title",""),
                        "direct_cause_details": r.get("direct_cause_details",""),
                        "day_of_created_date": r.get("day_of_created_date",""),
                    } for r in records
                ],
            })

    payload = {
        "schema_version": "1.1-hybrid",
        "generated_by": "scripts/generate_trends.py",
        "llm_used": llm_available(),
        "groups": groups_out,
    }
    return payload

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input .xlsx")
    ap.add_argument("--output", required=True, help="Path to output trends.json")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = first_visible_sheet(in_path)
    df = map_headers(df_raw)

    payload = make_payload(df)

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path.as_posix()} (llm_used={payload['llm_used']})")

if __name__ == "__main__":
    main()
