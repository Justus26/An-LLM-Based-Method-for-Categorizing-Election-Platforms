"""
Schritt 4: Klassifikation der Thueringer Programme mit ManifestoBERTa als
zweitem, unabhaengigem Klassifikator.

Da fuer Thueringen kein Goldstandard existiert, dient ManifestoBERTa nicht
der Fehlermessung, sondern der Robustheitspruefung: Aehnliche Positionen
bei zwei methodisch verschiedenen Klassifikatoren sprechen fuer die
Belastbarkeit der Ergebnisse. ManifestoBERTa nutzt keinen Beispiel-Pool
und haengt daher nicht vom Pool-Modus ab.

Kontext: ein Satz davor und einer danach, wie bei der LLM-Klassifikation.

Eingabe  data/thueringen/segmente_en.csv
Aufruf   python scripts/04_thueringen_manifestoberta.py
         python scripts/04_thueringen_manifestoberta.py --compare results/thueringen/thueringen_llm_fixed_2021.csv
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (DATA_DIR, RESULTS_DIR, atomic_write_json, context_string,
                    load_mapping, norm_code, position, run_manifestoberta)

INPUT_CSV = os.path.join(DATA_DIR, "thueringen", "segmente_en.csv")
OUT_DIR = os.path.join(RESULTS_DIR, "thueringen")
RUN_NAME = "thueringen_manifestoberta"
MIN_N = 30


def load_segments():
    df = pd.read_csv(INPUT_CSV, dtype={'year': str, 'party_id': str})
    junk = ~df['text_de'].astype(str).apply(lambda t: any(ch.isalpha() for ch in t))
    missing = df['text_en'].isna() | (df['text_en'].astype(str).str.strip() == '')
    df = df[~junk & ~missing].copy()
    for col in ('prev_en', 'next_en'):
        df[col] = df[col].fillna('') if col in df else ''
    print(f"[OK] {len(df)} Segmente")
    return df


def compare(df, llm_csv, mapping):
    """Uebereinstimmung mit der LLM-Klassifikation (kein Fehlermass)."""
    llm = pd.read_csv(llm_csv, dtype={'year': str})
    m = df.merge(llm[['manifesto_id', 'row_idx', 'top1']].astype({'top1': str}),
                 on=['manifesto_id', 'row_idx'], how='inner')
    if m.empty:
        return None
    m['sp_llm'] = [mapping.get(norm_code(c), '') for c in m['top1']]
    m['sp_berta'] = [mapping.get(norm_code(c), '') for c in m['berta_top1']]
    axis = m[m['sp_llm'].isin(['GAL', 'TAN', 'LEFT', 'RIGHT'])]
    diffs = []
    for _, g in m.groupby(['year', 'party']):
        a = position(g['top1'].astype(str), mapping, 'galtan')[0]
        b = position(g['berta_top1'], mapping, 'galtan')[0]
        if not (np.isnan(a) or np.isnan(b)):
            diffs.append(abs(a - b))
    return {'n_gemeinsam': len(m),
            'kategorie_uebereinstimmung': float((m['top1'] == m['berta_top1']).mean()),
            'achsen_uebereinstimmung': float((axis['sp_llm'] == axis['sp_berta']).mean()),
            'mittlere_abweichung_galtan': float(np.mean(diffs)) if diffs else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare", help="CSV-Ausgabe von 03_thueringen_llm.py")
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    mapping = load_mapping()
    df = load_segments()
    ctx = [context_string(p, t, n) for p, t, n in zip(df['prev_en'], df['text_en'], df['next_en'])]
    ranked = run_manifestoberta(list(df['text_en']), ctx)
    df['berta_ranked'] = ranked
    df['berta_top1'] = [r[0] if r else None for r in ranked]
    df['spektrum'] = [mapping.get(norm_code(c), '') if c else '' for c in df['berta_top1']]

    positions = []
    for (year, party), g in df.groupby(['year', 'party']):
        gt, gt_n = position(g['berta_top1'], mapping, 'galtan')
        lr, lr_n = position(g['berta_top1'], mapping, 'lr')
        positions.append({'year': year, 'party': party, 'n_segmente': len(g),
                          'galtan': None if gt != gt else round(gt, 4), 'n_galtan': gt_n,
                          'lr': None if lr != lr else round(lr, 4), 'n_lr': lr_n,
                          'unter_mindest_n': min(gt_n, lr_n) < MIN_N})

    comparison = compare(df, args.compare, mapping) if args.compare else None

    out = df[['manifesto_id', 'year', 'party', 'party_id', 'row_idx', 'text_de',
              'text_en', 'berta_top1', 'berta_ranked', 'spektrum']].copy()
    out['berta_ranked'] = out['berta_ranked'].apply(json.dumps)
    out.to_csv(os.path.join(OUT_DIR, f"{RUN_NAME}.csv"), index=False, encoding='utf-8')
    atomic_write_json({'config': {'n_segmente': len(df), 'mindest_n': MIN_N},
                       'positionen': positions, 'vergleich_mit_llm': comparison},
                      os.path.join(OUT_DIR, f"{RUN_NAME}.json"))
    print(f"[OK] Ergebnisse in {OUT_DIR}")


if __name__ == "__main__":
    main()
