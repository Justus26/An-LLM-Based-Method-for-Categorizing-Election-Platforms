"""
Schritt 2: Modellvergleich auf handkodierten Bundestagswahlprogrammen.

Vergleicht die LLM-Klassifikation (Few-Shot mit dynamisch abgerufenen
Beispielen) und ManifestoBERTa mit der Handkodierung des Manifesto
Project (Goldstandard) ueber mehrere Wahljahre. Ausgewertet werden
Kategorien (Top-1/2/3, Precision, Recall, F1) sowie die daraus
abgeleiteten Parteipositionen auf GAL-TAN und Links-Rechts.

Pool-Modi
  fixed_2021   ein Beispiel-Pool (Bundestagswahl 2021) fuer alle Testjahre
  era_matched  je Testjahr der Pool mit gleicher Handbuchversion, sonst der
               naechstliegende (Pool 2002 fuer Handbuch 2, Pool 2021 fuer
               Handbuch 4 und 5)
  Die Pool-Jahre sind keine Testjahre, sonst waere es Leakage.

Stichprobe  je Partei und Wahljahr bis zu 200 Saetze, damit kleine Parteien
            die Positionsschaetzung nicht verzerren.

Daten
  Abruf ueber die Manifesto-API (MANIFESTO_API_KEY in .env). Jahre, die dort
  noch nicht verfuegbar sind, als CSV in data/manifestos_local/ ablegen:
  Dateiname <party_id>_<YYYYMM>.csv mit den Spalten text, text_en, cmp_code.

Aufruf
  python scripts/02_haupttest_bundestag.py --pool-mode fixed_2021
  python scripts/02_haupttest_bundestag.py --pool-mode era_matched
  Option --no-berta ueberspringt ManifestoBERTa (es haengt nicht vom Pool ab).
"""

import argparse
import glob
import io
import json
import os
import random
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from common import (DATA_DIR, MAX_WORKERS, RANDOM_SEED, RESULTS_DIR,
                    atomic_write_json, classification_metrics, classify,
                    context_string, embed_texts, load_code_list, load_mapping,
                    norm_code, pool_from_statements, pool_paths, position,
                    retrieve, run_manifestoberta, system_prompt)

# ═══════════════════════════════════════════════════════════════════════
# KONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

TEST_YEARS = ["199809", "200509", "200909", "201309", "201709", "202502"]
POOL_CANDIDATE_YEARS = ["200209", "202109"]
POOL_SIZE = 5900
SAMPLE_PER_PARTY = 200

# Handbuchversion je Wahljahr (Manifesto-Variable 'manual')
HANDBOOK_BY_YEAR = {
    "199809": 2, "200209": 2, "200509": 2, "200909": 2,
    "201309": 4, "201709": 5, "202109": 5, "202502": 5,
}

# Parteien ohne Bezug zur Fragestellung (Piratenpartei)
EXCLUDE_PARTY_IDS = {"41950", "41952"}

PARTY_NAMES = {
    '41111': 'Grüne', '41112': 'Grüne', '41113': 'Grüne',
    '41221': 'PDS', '41222': 'Linkspartei.PDS', '41223': 'Die Linke',
    '41320': 'SPD', '41420': 'FDP', '41440': 'FW',
    '41521': 'CDU/CSU', '41523': 'CDU/CSU',
    '41701': 'REP', '41702': 'NPD', '41703': 'DVU',
    '41730': 'BSW', '41912': 'SSW', '41950': 'Piraten', '41953': 'AfD',
}

MANIFESTO_CORE_API = "https://manifesto-project.wzb.eu/tools/api_get_core.json"
MANIFESTO_METADATA_API = "https://manifesto-project.wzb.eu/api/v1/metadata"
MANIFESTO_TEXTS_API = "https://manifesto-project.wzb.eu/api/v1/texts_and_annotations"
MANIFESTO_VERSIONS_API = "https://manifesto-project.wzb.eu/api/v1/list_metadata_versions"
MANIFESTO_CORE_VERSION = os.getenv("MANIFESTO_CORE_VERSION", "MPDS2024a")
MANIFESTO_CORPUS_VERSION = os.getenv("MANIFESTO_CORPUS_VERSION")  # leer = neueste

LOCAL_DIR = os.path.join(DATA_DIR, "manifestos_local")
OUT_DIR = os.path.join(RESULTS_DIR, "haupttest")
TEST_CORPUS_CACHE = os.path.join(OUT_DIR, "test_corpus.json")
TEST_EMB_CACHE = os.path.join(OUT_DIR, "test_embeddings.pkl")


# ═══════════════════════════════════════════════════════════════════════
# DATEN
# ═══════════════════════════════════════════════════════════════════════

def load_local_year(date_key):
    stmts = []
    for path in sorted(glob.glob(os.path.join(LOCAL_DIR, f"*_{date_key}.csv"))):
        pid = os.path.basename(path).split('_')[0]
        if pid in EXCLUDE_PARTY_IDS:
            continue
        df = pd.read_csv(path).reset_index(drop=True)
        for idx, row in df.iterrows():
            gt = norm_code(row.get('cmp_code'))
            text = row.get('text_en')
            if gt is None or pd.isna(text) or len(str(text)) < 5:
                continue
            prev_t = df.loc[idx - 1, 'text_en'] if idx > 0 else ""
            next_t = df.loc[idx + 1, 'text_en'] if idx < len(df) - 1 else ""
            stmts.append({'year': date_key, 'party': PARTY_NAMES.get(pid, pid),
                          'party_id': pid, 'row_idx': int(idx), 'text': str(text),
                          'ground_truth': gt,
                          'prev_context': "" if pd.isna(prev_t) else str(prev_t),
                          'next_context': "" if pd.isna(next_t) else str(next_t)})
    return stmts


def get_corpus_version(api_key):
    if MANIFESTO_CORPUS_VERSION:
        return MANIFESTO_CORPUS_VERSION
    r = requests.get(MANIFESTO_VERSIONS_API, params={'api_key': api_key}, timeout=60)
    r.raise_for_status()
    versions = r.json().get('versions', [])
    if not versions:
        raise SystemExit("[STOP] Keine Korpus-Versionen erhalten - API-Schluessel pruefen.")
    print(f"[OK] Korpus-Version: {versions[-1]}")
    return versions[-1]


def fetch_year(date_key, api_key, corpus_version):
    """Kerndatensatz -> Metadaten -> annotierte Texte (englische Uebersetzung)."""
    core = requests.get(MANIFESTO_CORE_API, params={
        'api_key': api_key, 'key': MANIFESTO_CORE_VERSION, 'raw': 'true'}, timeout=120)
    core.raise_for_status()
    core_df = pd.read_csv(io.StringIO(core.text), dtype=str)
    sel = core_df[(core_df['country'].str.strip() == '41')
                  & (core_df['date'].str.strip() == date_key)]
    keys = [f"{str(p).strip()}_{date_key}" for p in sel['party']]
    if not keys:
        return []

    meta = requests.get(MANIFESTO_METADATA_API, params=[
        ('api_key', api_key), ('version', corpus_version)]
        + [('keys[]', k) for k in keys], timeout=120)
    meta.raise_for_status()

    usable = []
    for it in meta.json().get('items', []):
        pid, mid = str(it.get('party_id', '')).strip(), it.get('manifesto_id')
        if not mid or not pid or pid in EXCLUDE_PARTY_IDS:
            continue
        if str(it.get('annotations')).lower() == 'false':
            continue
        usable.append((pid, mid))

    stmts = []
    for pid, mid in usable:
        resp = requests.get(MANIFESTO_TEXTS_API, params={
            'api_key': api_key, 'keys[]': mid, 'version': corpus_version,
            'translation': 'en'}, timeout=120)
        if resp.status_code != 200:
            print(f"     [!] {mid}: HTTP {resp.status_code}")
            continue
        items = resp.json().get('items', [])
        if not items or 'items' not in items[0]:
            continue
        rowlist = items[0]['items']
        for i, it in enumerate(rowlist):
            gt = norm_code(it.get('cmp_code'))
            text = it.get('text', '')
            if gt is None or len(str(text)) < 5:
                continue
            stmts.append({
                'year': date_key, 'party': PARTY_NAMES.get(pid, pid),
                'party_id': pid, 'row_idx': i, 'text': str(text), 'ground_truth': gt,
                'prev_context': str(rowlist[i - 1].get('text', '')) if i > 0 else "",
                'next_context': (str(rowlist[i + 1].get('text', ''))
                                 if i < len(rowlist) - 1 else "")})
    return stmts


def looks_english(texts, n=200):
    """Die API liefert ohne Uebersetzung stillschweigend das Original."""
    rng = random.Random(RANDOM_SEED)
    sample = rng.sample(list(texts), min(n, len(texts)))
    markers = (" und ", " der ", " die ", " das ", " nicht ", " wir ")
    share = np.mean([any(m in f" {t.lower()} " for m in markers) for t in sample])
    return share < 0.2


def build_test_set(api_key):
    if os.path.exists(TEST_CORPUS_CACHE):
        with open(TEST_CORPUS_CACHE, encoding='utf-8') as f:
            per_year = json.load(f)
    else:
        version = get_corpus_version(api_key)
        per_year = {}
        for y in TEST_YEARS:
            local = load_local_year(y)
            stmts = local if local else fetch_year(y, api_key, version)
            print(f"[i] {y}: {len(stmts)} kodierte Saetze")
            if not stmts:
                continue
            if not looks_english([s['text'] for s in stmts]):
                raise SystemExit(f"[STOP] {y}: keine englische Uebersetzung verfuegbar.")
            per_year[y] = stmts
        atomic_write_json(per_year, TEST_CORPUS_CACHE)

    rng = random.Random(RANDOM_SEED)
    test = []
    for y in TEST_YEARS:
        by_party = defaultdict(list)
        for s in per_year.get(y, []):
            if str(s.get('party_id')) not in EXCLUDE_PARTY_IDS:
                by_party[s['party_id']].append(s)
        for _, group in sorted(by_party.items()):
            test.extend(group if len(group) <= SAMPLE_PER_PARTY
                        else rng.sample(group, SAMPLE_PER_PARTY))
    print(f"[OK] Testset: {len(test)} Saetze")
    return test


# ═══════════════════════════════════════════════════════════════════════
# POOLS
# ═══════════════════════════════════════════════════════════════════════

def build_pool(pool_year, api_key, corpus_version):
    """Pool aufbauen oder aus data/pools/ laden und einbetten."""
    corpus_path, emb_path = pool_paths(pool_year)
    if os.path.exists(corpus_path):
        with open(corpus_path, encoding='utf-8') as f:
            stmts = json.load(f)
    else:
        print(f"[..] Pool {pool_year} wird abgerufen")
        stmts = fetch_year(pool_year, api_key, corpus_version)
        atomic_write_json(stmts, corpus_path)
    rng = random.Random(RANDOM_SEED)
    if len(stmts) > POOL_SIZE:
        stmts = rng.sample(stmts, POOL_SIZE)
    emb = embed_texts([s['text'] for s in stmts], emb_path)
    pool, mat = pool_from_statements(stmts, emb)
    print(f"[OK] Pool {pool_year}: {len(pool)} Saetze")
    return pool, mat


def pool_for_test_year(test_year):
    cands = [y for y in POOL_CANDIDATE_YEARS if y != test_year]

    def dist(y):
        return abs(int(y[:4]) - int(test_year[:4]))

    hb = HANDBOOK_BY_YEAR.get(test_year)
    if hb is None:
        return min(cands, key=dist)
    same = [y for y in cands if HANDBOOK_BY_YEAR.get(y) == hb]
    if same:
        return min(same, key=dist)
    return min(cands, key=lambda y: (abs(HANDBOOK_BY_YEAR.get(y, 99) - hb), dist(y)))


def build_neighbours(test, test_emb, pool_mode, api_key):
    """Beispiele je (Jahr, Text) - so koennen gleiche Saetze in Jahren mit
    unterschiedlichen Pools unterschiedliche Beispiele erhalten."""
    years = sorted({s['year'] for s in test})
    if pool_mode == "fixed_2021":
        assign = {y: "202109" for y in years}
    else:
        assign = {y: pool_for_test_year(y) for y in years}
    version = get_corpus_version(api_key)
    pools = {py: build_pool(py, api_key, version) for py in sorted(set(assign.values()))}
    nb = {}
    for y in years:
        pool, mat = pools[assign[y]]
        texts = [s['text'] for s in test if s['year'] == y]
        for t, ex in retrieve(texts, test_emb, pool, mat).items():
            nb[(y, t)] = ex
    return nb, assign


# ═══════════════════════════════════════════════════════════════════════
# KLASSIFIKATION
# ═══════════════════════════════════════════════════════════════════════

def key_of(s):
    return (s['year'], str(s.get('party_id', '')), int(s['row_idx']))


def run_llm(test, nb, pool_mode, checkpoint_path):
    code_list, valid = load_code_list()
    sys_p = system_prompt(code_list)
    results = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding='utf-8') as f:
            for r in json.load(f):
                if r.get('pool_mode') == pool_mode and r.get('ranked'):
                    results[key_of(r)] = r
    todo = [s for s in test if key_of(s) not in results and (s['year'], s['text']) in nb]
    print(f"[..] LLM ({pool_mode}): {len(todo)} offen, {len(results)} erledigt")

    def work(s):
        ranked = classify(s['text'], nb[(s['year'], s['text'])], sys_p, valid,
                          s.get('prev_context', ''), s.get('next_context', ''))
        return {'year': s['year'], 'party_id': s['party_id'], 'row_idx': s['row_idx'],
                'pool_mode': pool_mode, 'ranked': ranked or []}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(work, s) for s in todo]
        for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures), desc="LLM")):
            r = fut.result()
            if r['ranked']:
                results[key_of(r)] = r
            if (i + 1) % 200 == 0:
                atomic_write_json(list(results.values()), checkpoint_path)
    atomic_write_json(list(results.values()), checkpoint_path)
    return {k: r['ranked'] for k, r in results.items()}


# ═══════════════════════════════════════════════════════════════════════
# AUSWERTUNG
# ═══════════════════════════════════════════════════════════════════════

def positions_vs_gold(df, mapping, col):
    """Rangkorrelation (Spearman) und mittlerer absoluter Fehler (MAE) der
    Parteipositionen gegenueber Gold, je Wahljahr und Achse."""
    out = []
    for year, g in df.groupby('year'):
        gold, pred = {}, {}
        for party, pg in g.groupby('party'):
            gold[party] = {ax: position(pg['ground_truth'], mapping, ax)[0]
                           for ax in ('galtan', 'lr')}
            pred[party] = {ax: position(pg[col], mapping, ax)[0]
                           for ax in ('galtan', 'lr')}
        parties = [p for p in gold if all(not np.isnan(gold[p][a]) and not np.isnan(pred[p][a])
                                          for a in ('galtan', 'lr'))]
        rec = {'year': year, 'n_parties': len(parties)}
        for ax in ('galtan', 'lr'):
            gv = np.array([gold[p][ax] for p in parties])
            pv = np.array([pred[p][ax] for p in parties])
            rec[f'mae_{ax}'] = float(np.mean(np.abs(gv - pv))) if parties else None
            rec[f'rho_{ax}'] = (float(spearmanr(gv, pv).statistic)
                                if len(parties) > 2 else None)
        out.append(rec)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-mode", choices=["fixed_2021", "era_matched"],
                        default="fixed_2021")
    parser.add_argument("--no-berta", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("MANIFESTO_API_KEY")
    if not api_key:
        raise SystemExit("MANIFESTO_API_KEY fehlt - siehe .env.example")
    os.makedirs(OUT_DIR, exist_ok=True)
    run_name = f"haupttest_{args.pool_mode}"

    test = build_test_set(api_key)
    test_emb = embed_texts([s['text'] for s in test], TEST_EMB_CACHE)
    nb, assign = build_neighbours(test, test_emb, args.pool_mode, api_key)
    llm = run_llm(test, nb, args.pool_mode,
                  os.path.join(OUT_DIR, f"{run_name}_checkpoint.json"))

    rows = []
    for s in test:
        ranked = llm.get(key_of(s))
        if ranked:
            rows.append({'year': s['year'], 'party': s['party'], 'party_id': s['party_id'],
                         'row_idx': s['row_idx'], 'text': s['text'],
                         'prev_context': s.get('prev_context', ''),
                         'next_context': s.get('next_context', ''),
                         'ground_truth': s['ground_truth'],
                         'llm_ranked': ranked, 'llm_top1': ranked[0]})
    df = pd.DataFrame(rows)
    print(f"[OK] {len(df)} von {len(test)} Saetzen klassifiziert")

    if not args.no_berta:
        ctx = [context_string(p, t, n) for p, t, n in
               zip(df['prev_context'], df['text'], df['next_context'])]
        berta = run_manifestoberta(list(df['text']), ctx)
        df['berta_ranked'] = berta
        df['berta_top1'] = [r[0] if r else None for r in berta]

    mapping = load_mapping()
    summary = {'overall': {}, 'per_year': {}}
    systems = [('llm', 'llm_ranked')] + ([] if args.no_berta else [('manifestoberta', 'berta_ranked')])
    for name, col in systems:
        summary['overall'][name] = classification_metrics(list(df['ground_truth']), list(df[col]))
        summary['per_year'][name] = {
            y: classification_metrics(list(g['ground_truth']), list(g[col]))
            for y, g in df.groupby('year')}

    positions = {'llm': positions_vs_gold(df, mapping, 'llm_top1')}
    if not args.no_berta:
        positions['manifestoberta'] = positions_vs_gold(df, mapping, 'berta_top1')

    for name, col in [('gold', 'ground_truth'), ('llm', 'llm_top1')] + \
            ([] if args.no_berta else [('berta', 'berta_top1')]):
        df[f'{name}_spektrum'] = [mapping.get(norm_code(c), '') if c else '' for c in df[col]]

    out = df.drop(columns=['prev_context', 'next_context']).copy()
    for col in ('llm_ranked', 'berta_ranked'):
        if col in out:
            out[col] = out[col].apply(json.dumps)
    out.to_csv(os.path.join(OUT_DIR, f"{run_name}.csv"), index=False, encoding='utf-8')
    atomic_write_json({'config': {'pool_mode': args.pool_mode, 'pool_by_year': assign,
                                  'test_years': TEST_YEARS,
                                  'sample_per_party': SAMPLE_PER_PARTY,
                                  'seed': RANDOM_SEED},
                       'metrics': summary, 'positions': positions},
                      os.path.join(OUT_DIR, f"{run_name}.json"))

    o = summary['overall']['llm']
    print(f"\nLLM: Top-1 {o['top1']:.1%}, Top-3 {o['top3']:.1%}, F1 (gewichtet) {o['f1_weighted']:.3f}")
    if not args.no_berta:
        b = summary['overall']['manifestoberta']
        print(f"ManifestoBERTa: Top-1 {b['top1']:.1%}, Top-3 {b['top3']:.1%}, "
              f"F1 (gewichtet) {b['f1_weighted']:.3f}")
    print(f"[OK] Ergebnisse in {OUT_DIR}")


if __name__ == "__main__":
    main()
