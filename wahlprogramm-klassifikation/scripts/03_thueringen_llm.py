"""
Schritt 3: Klassifikation der Thueringer Landtagswahlprogramme (1990-2024)
mit dem LLM und Berechnung der Parteipositionen.

Fuer Thueringen gibt es keinen Goldstandard. Ausgegeben werden deshalb
Positionen je Partei und Wahljahr sowie Plausibilitaetsangaben
(Achsenverteilung, haeufigste Kategorien, Zahl achsenrelevanter Saetze).
Positionen mit weniger als 30 achsenrelevanten Saetzen werden markiert.

Pool-Modi
  fixed_2021   ein Pool (Bundestag 2021, Handbuch 5) fuer alle Jahrgaenge -
               einheitliche Kodierkonvention ueber die gesamte Zeitreihe
  era_matched  Pool 2002 (Handbuch 2) fuer 1990-2009, Pool 2021 ab 2014
Beide Pools muessen vorher von 02_haupttest_bundestag.py erzeugt worden sein.

Eingabe  data/thueringen/segmente_en.csv (Ausgabe von 01_uebersetzung.py)
Aufruf   python scripts/03_thueringen_llm.py --pool-mode fixed_2021
"""

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from common import (DATA_DIR, MAX_WORKERS, RESULTS_DIR, atomic_write_json,
                    classify, embed_texts, load_code_list, load_mapping,
                    load_pool, norm_code, position, retrieve, system_prompt)

INPUT_CSV = os.path.join(DATA_DIR, "thueringen", "segmente_en.csv")
EMB_CACHE = os.path.join(DATA_DIR, "thueringen", "segmente_embeddings.pkl")
OUT_DIR = os.path.join(RESULTS_DIR, "thueringen")
MIN_N = 30

HANDBOOK_BY_YEAR = {'1990': 2, '1994': 2, '1999': 2, '2004': 2, '2009': 2,
                    '2014': 5, '2019': 5, '2024': 5}
POOL_BY_HANDBOOK = {2: '200209', 5: '202109'}


def load_segments():
    df = pd.read_csv(INPUT_CSV, dtype={'year': str, 'party_id': str})
    # Artefakte der PDF-Konvertierung (Segmente ohne Buchstaben) entfernen
    junk = ~df['text_de'].astype(str).apply(lambda t: any(ch.isalpha() for ch in t))
    missing = df['text_en'].isna() | (df['text_en'].astype(str).str.strip() == '')
    if junk.any() or missing.any():
        print(f"[i] entfernt: {int(junk.sum())} Artefakte, "
              f"{int((missing & ~junk).sum())} ohne Uebersetzung")
    df = df[~junk & ~missing].copy()
    for col in ('prev_en', 'next_en'):
        df[col] = df[col].fillna('') if col in df else ''
    print(f"[OK] {len(df)} Segmente")
    return df.to_dict('records')


def seg_key(r):
    return (str(r['manifesto_id']), int(r['row_idx']))


def pool_for_year(year, pool_mode):
    if pool_mode == 'fixed_2021':
        return '202109'
    return POOL_BY_HANDBOOK[HANDBOOK_BY_YEAR[str(year)[:4]]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-mode", choices=["fixed_2021", "era_matched"],
                        default="fixed_2021")
    args = parser.parse_args()
    run_name = f"thueringen_llm_{args.pool_mode}"
    os.makedirs(OUT_DIR, exist_ok=True)
    checkpoint = os.path.join(OUT_DIR, f"{run_name}_checkpoint.json")

    code_list, valid = load_code_list()
    sys_p = system_prompt(code_list)
    mapping = load_mapping()
    segments = load_segments()

    assign = {y: pool_for_year(y, args.pool_mode) for y in sorted({s['year'] for s in segments})}
    text_emb = embed_texts([s['text_en'] for s in segments], EMB_CACHE)
    pools = {py: load_pool(py) for py in sorted(set(assign.values()))}

    # Beispiele je (Jahr, Text): identische Textbausteine in verschiedenen
    # Jahren koennen so aus unterschiedlichen Pools bedient werden.
    nb = {}
    for y, py in assign.items():
        pool, mat = pools[py]
        texts = [s['text_en'] for s in segments if s['year'] == y]
        for t, ex in retrieve(texts, text_emb, pool, mat).items():
            nb[(y, t)] = ex

    results = {}
    if os.path.exists(checkpoint):
        with open(checkpoint, encoding='utf-8') as f:
            results = {seg_key(r): r for r in json.load(f) if r.get('ranked')}
    todo = [s for s in segments if seg_key(s) not in results and (s['year'], s['text_en']) in nb]
    print(f"[..] Klassifikation: {len(todo)} offen, {len(results)} erledigt")

    def work(s):
        ranked = classify(s['text_en'], nb[(s['year'], s['text_en'])], sys_p, valid,
                          s.get('prev_en', ''), s.get('next_en', ''))
        return {**{k: s.get(k) for k in ('manifesto_id', 'year', 'party', 'party_id',
                                         'row_idx', 'text_de', 'text_en')},
                'ranked': ranked or []}

    n_fail = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(work, s) for s in todo]
        for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures),
                                     desc="Klassifikation")):
            r = fut.result()
            if r['ranked']:
                results[seg_key(r)] = r
            else:
                n_fail += 1
            if (i + 1) % 250 == 0:
                atomic_write_json(list(results.values()), checkpoint)
    atomic_write_json(list(results.values()), checkpoint)
    if n_fail:
        print(f"[!] {n_fail} Saetze ohne Ergebnis - beim naechsten Start erneut versucht")

    df = pd.DataFrame(results.values())
    df['top1'] = df['ranked'].apply(lambda r: r[0])
    df['spektrum'] = [mapping.get(norm_code(c), '') for c in df['top1']]

    spectrum_share = df['spektrum'].value_counts(normalize=True).round(4).to_dict()
    top_categories = df['top1'].value_counts(normalize=True).head(10).round(4).to_dict()
    positions = []
    for (year, party), g in df.groupby(['year', 'party']):
        gt, gt_n = position(g['top1'], mapping, 'galtan')
        lr, lr_n = position(g['top1'], mapping, 'lr')
        positions.append({'year': year, 'party': party, 'n_segmente': len(g),
                          'galtan': None if gt != gt else round(gt, 4), 'n_galtan': gt_n,
                          'lr': None if lr != lr else round(lr, 4), 'n_lr': lr_n,
                          'unter_mindest_n': min(gt_n, lr_n) < MIN_N})

    out = df.copy()
    out['ranked'] = out['ranked'].apply(json.dumps)
    out.to_csv(os.path.join(OUT_DIR, f"{run_name}.csv"), index=False, encoding='utf-8')
    atomic_write_json({'config': {'pool_mode': args.pool_mode, 'pool_by_year': assign,
                                  'n_segmente': len(df), 'mindest_n': MIN_N},
                       'achsenverteilung': spectrum_share,
                       'haeufigste_kategorien': top_categories,
                       'positionen': positions},
                      os.path.join(OUT_DIR, f"{run_name}.json"))
    print(f"[OK] Ergebnisse in {OUT_DIR}")


if __name__ == "__main__":
    main()
