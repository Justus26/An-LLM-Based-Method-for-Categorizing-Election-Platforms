"""
Validierung der Segmentierung gegen die Handsegmentierung des Manifesto
Project.

Dasselbe Wahlprogramm (FDP, Bundestagswahl 2025) liegt zweimal vor: als
Rohtext bei PolDoc und als von Hand in Quasi-Saetze zerlegte und kodierte
Fassung beim Manifesto Project. Die eigene Segmentierung des Rohtexts wird
mit der Handsegmentierung verglichen.

Einstufung je eigenem Segment
  EXAKT     nach Normalisierung identisch mit einem Quasi-Satz des Goldstandards
  AEHNLICH  Textaehnlichkeit >= SIMILARITY_THRESHOLD oder Teilstring
  KEIN      nichts Vergleichbares gefunden
Zusaetzlich: Anteil der Gold-Quasi-Saetze ohne Entsprechung (Hinweis auf
zusammengezogene Saetze).

Einschraenkung: Abgleich an genau einem Dokument. Das zeigt die
Groessenordnung der Uebereinstimmung, sichert sie aber nicht statistisch ab.

Eingabe
  data/validierung/41420.000.2025.1.1.txt   Rohtext (PolDoc)
  data/manifestos_local/41420_202502.csv    Goldstandard (Spalten text, cmp_code)
Ausgabe
  results/segmentierung/

Aufruf   python scripts/00b_segmentierung_validierung.py
"""

import difflib
import os
import re
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA_DIR, RESULTS_DIR, atomic_write_json, norm_code
from segmentation import load_nlp, segment

RAW_TEXT = os.path.join(DATA_DIR, "validierung", "41420.000.2025.1.1.txt")
GOLD_CSV = os.path.join(DATA_DIR, "manifestos_local", "41420_202502.csv")
OUT_DIR = os.path.join(RESULTS_DIR, "segmentierung")

SIMILARITY_THRESHOLD = 0.80


def normalise(t):
    t = t.lower().strip()
    t = re.sub(r'[„"“”‚‘’]', '', t)
    t = re.sub(r'[-–—]', ' ', t)
    t = re.sub(r'[^\wäöüß ]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def load_gold():
    """Deutsche Quasi-Saetze; Ueberschriften und nicht kodierbare
    Subkategorien werden wie in der uebrigen Pipeline ausgeschlossen."""
    d = pd.read_csv(GOLD_CSV)
    d = d[d['text'].notna() & (d['text'].astype(str).str.len() >= 5)]
    d = d[d['cmp_code'].apply(lambda c: norm_code(c) is not None)]
    return list(d['text'].astype(str))


def best_match(text, candidates):
    best_ratio, best_idx = 0.0, -1
    for i, c in enumerate(candidates):
        r = difflib.SequenceMatcher(None, text, c).ratio()
        if r > best_ratio:
            best_ratio, best_idx = r, i
    contains = False
    if best_idx >= 0:
        a, b = text, candidates[best_idx]
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        contains = len(shorter) > 15 and shorter in longer
    return best_idx, best_ratio, contains


def evaluate(own, gold):
    own_norm = [normalise(s) for s in own]
    gold_norm = [normalise(s) for s in gold]
    gold_index = {}
    for j, g in enumerate(gold_norm):
        gold_index.setdefault(g, j)   # erstes Vorkommen, wie list.index()
    gold_hit, results = set(), []
    for seg_raw, seg in zip(own, own_norm):
        if seg in gold_index:
            j = gold_index[seg]
            results.append({'segment': seg_raw, 'match': 'EXAKT', 'gold': gold[j]})
            gold_hit.add(j)
            continue
        j, ratio, contains = best_match(seg, gold_norm)
        if ratio >= SIMILARITY_THRESHOLD or contains:
            results.append({'segment': seg_raw, 'match': 'AEHNLICH', 'gold': gold[j]})
            gold_hit.add(j)
        else:
            results.append({'segment': seg_raw, 'match': 'KEIN', 'gold': None})
    missed = [g for j, g in enumerate(gold) if j not in gold_hit]
    return results, missed


def main():
    for p in (RAW_TEXT, GOLD_CSV):
        if not os.path.exists(p):
            raise SystemExit(f"{p} fehlt.")
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(RAW_TEXT, encoding='utf-8', errors='ignore') as f:
        own = segment(f.read(), load_nlp())
    gold = load_gold()
    results, missed = evaluate(own, gold)

    cnt = Counter(r['match'] for r in results)
    n = len(results)
    summary = {'eigene_segmente': n, 'gold_quasisaetze': len(gold),
               **{k.lower(): cnt[k] / n for k in ('EXAKT', 'AEHNLICH', 'KEIN')},
               'gold_ohne_entsprechung': len(missed) / len(gold)}
    for k in ('EXAKT', 'AEHNLICH', 'KEIN'):
        print(f"  {k:<10}{cnt[k]:>5}  ({cnt[k] / n:.1%})")
    print(f"  Gold ohne Entsprechung: {len(missed)} von {len(gold)} "
          f"({len(missed) / len(gold):.1%})")

    pd.DataFrame(results).to_csv(os.path.join(OUT_DIR, "abgleich_segmente.csv"),
                                 index=False, encoding='utf-8')
    pd.DataFrame({'gold_ohne_entsprechung': missed}).to_csv(
        os.path.join(OUT_DIR, "gold_ohne_entsprechung.csv"), index=False, encoding='utf-8')
    atomic_write_json(summary, os.path.join(OUT_DIR, "zusammenfassung.json"))
    print(f"[OK] Ergebnisse in {OUT_DIR}")


if __name__ == "__main__":
    main()
