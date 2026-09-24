"""
Gemeinsame Funktionen fuer alle Skripte dieses Repositorys.

Enthaelt: Konfiguration ueber Umgebungsvariablen, Kategorienliste und
Prompt, Klassifikation per LLM, Embeddings und Retrieval, Laden der
Beispiel-Pools, ManifestoBERTa, GAL-TAN/Links-Rechts-Mapping sowie
Kennzahlen.

Alle Zugangsdaten werden aus einer Datei .env im Projektordner gelesen
(Vorlage: .env.example). Es sind keine Schluessel im Code hinterlegt.
"""

import json
import os
import pickle
import time
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ═══════════════════════════════════════════════════════════════════════
# PFADE
# ═══════════════════════════════════════════════════════════════════════

CONFIG_DIR = "config"
DATA_DIR = "data"
POOL_DIR = os.path.join(DATA_DIR, "pools")
RESULTS_DIR = "results"

CATEGORIES_CSV = os.path.join(CONFIG_DIR, "categories.csv")
MAPPING_CSV = os.path.join(CONFIG_DIR, "gal_tan_left_right_mapping.csv")

# ═══════════════════════════════════════════════════════════════════════
# MODELLE UND PARAMETER
# ═══════════════════════════════════════════════════════════════════════

LLM_MODEL = os.getenv("LLM_MODEL", "gemma4-31b-it")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "octen-embedding-8b")
BERTA_MODEL = os.getenv(
    "BERTA_MODEL",
    "manifesto-project/manifestoberta-xlm-roberta-56policy-topics-context-2026-1-1")

TEMPERATURE = 0.1   # geringe Zufaelligkeit, Ergebnisse nicht exakt deterministisch
MAX_TOKENS = 200
TOP_K = 15          # Anzahl abgerufener Beispielsaetze je Satz
N_RANKED = 3        # Rangliste der drei wahrscheinlichsten Kategorien
RANDOM_SEED = 42

MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
MAX_RETRIES = 3
RETRY_DELAY = 2

# Subkategorien ohne Entsprechung in Handbuch 4 gelten als nicht kodierbar
EXCLUDED_SUBCODES = {'202.2', '605.2', '703.2'}

_client = None


def get_client():
    """OpenAI-kompatibler Client. LLM_BASE_URL leer lassen fuer die
    offizielle OpenAI-API, sonst die Adresse des eigenen Servers."""
    global _client
    if _client is None:
        import openai
        key = os.getenv("LLM_API_KEY")
        if not key:
            raise SystemExit("LLM_API_KEY fehlt - siehe .env.example")
        _client = openai.OpenAI(api_key=key,
                                base_url=os.getenv("LLM_BASE_URL") or None,
                                timeout=60.0, max_retries=0)
    return _client


# ═══════════════════════════════════════════════════════════════════════
# DATEIEN SICHER SCHREIBEN
# ═══════════════════════════════════════════════════════════════════════

def atomic_write_json(obj, path):
    """Erst in .tmp schreiben, dann ersetzen: ein Abbruch hinterlaesst nie
    eine halbe Datei."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def atomic_pickle_dump(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


# ═══════════════════════════════════════════════════════════════════════
# KATEGORIEN UND PROMPT
# ═══════════════════════════════════════════════════════════════════════

def norm_code(raw) -> Optional[str]:
    """Manifesto-Code auf die Hauptkategorie normalisieren ('201.1' -> '201')."""
    s = str(raw).strip()
    if s in EXCLUDED_SUBCODES:
        return None
    if not any(ch.isdigit() for ch in s):
        return None
    if '.' in s:
        s = s.split('.')[0]
    try:
        return str(int(s))
    except ValueError:
        return None


def load_code_list():
    """Kategorienliste (Spalten Code;Name) fuer den Prompt."""
    df = pd.read_csv(CATEGORIES_CSV, sep=';', encoding='utf-8-sig',
                     dtype={'Code': str}).fillna("")
    names, valid = {}, set()
    for _, row in df.iterrows():
        code = norm_code(row['Code'])
        if code is None:
            continue
        names[code] = str(row['Name']).strip()
        valid.add(code)
    listing = "\n".join(f"[{c}] {names[c]}"
                        for c in sorted(names, key=lambda x: int(x)))
    return listing, valid


# Regeln fuer haeufig verwechselte Kategorienpaare
RULES = """
Decision rules for frequently confused categories. Decide by the POSITION the statement takes, not by the topic it mentions:

1. 603 vs 604 (Traditional Morality). 603 = the statement defends or promotes traditional or religious values, the traditional family or religious heritage. This includes statements that criticise progressive policies in order to protect such values. 604 = the statement opposes traditional or religious values or wants to reduce their role.

2. 607 vs 608 (Multiculturalism). 608 = the statement rejects multiculturalism, warns against parallel societies, or demands that immigrants adopt the national culture, language or values. 607 = the statement favours cultural diversity and the preservation of minority cultures. The word "integration" does not make a statement 607 if it demands adaptation.

3. 107 vs 109 (Internationalism). 109 = the statement is sceptical of international organisations or cooperation, or puts national sovereignty above international commitments. 107 = the statement supports international cooperation, international organisations or development aid.

4. 108 vs 110 (European Union). 110 = the statement criticises the EU, wants to reduce its powers or return competences to the nation state. 108 = the statement supports the EU or deeper European integration.

5. 401 vs 403 (Markets). 401 = the statement favours free markets, private enterprise or less state intervention in the economy. 403 = the statement calls for rules for markets, such as competition law, consumer protection or price regulation.

6. 601 vs 605. 601 = national identity, national pride, protection of the national culture, or restricting immigration. 605 = crime, policing, courts and public security in general. A statement about limiting immigration is 601 unless its main point is crime.

7. 416 vs 501. 416 = the statement questions economic growth as a goal or calls for transforming the economy towards sustainability. 501 = protection of nature, climate or the environment without a focus on transforming the economy.
"""


def system_prompt(code_list):
    return f"""You are an expert political scientist classifying sentences from election manifestos according to the Manifesto Project coding scheme.

Available categories:
{code_list}

You are given examples of how human coders classified similar statements. Infer the coding conventions from these examples and apply them to the new statement.
{RULES}
Rank the {N_RANKED} most likely categories, best first.
Respond EXCLUSIVELY with valid JSON, no explanation.
Format: {{"ranked": ["<code1>", "<code2>", "<code3>"]}}"""


def user_prompt(text, examples, prev="", nxt=""):
    ex = "Examples of how human coders classified similar statements:\n"
    for e in examples:
        ex += f"- Code {e['ground_truth']}: {e['text']}\n"
    ctx = ""
    if prev:
        ctx += f"Prev: {prev}\n"
    if nxt:
        ctx += f"Next: {nxt}\n"
    return f"{ex}\n{ctx}Statement to classify:\n{text}\n\nJSON:"


def parse_ranked(out, valid_codes, n=N_RANKED):
    """Rangliste aus der Modellantwort lesen; faellt auf Ziffernsuche
    zurueck, falls das JSON unvollstaendig ist."""
    codes = []

    def add(raw):
        code = norm_code(raw)
        if code is not None and code in valid_codes and code not in codes:
            codes.append(code)

    try:
        parsed = json.loads(out[out.index('{'):out.rindex('}') + 1])
        for code in parsed.get('ranked', []):
            add(code)
    except (ValueError, json.JSONDecodeError, AttributeError):
        pass
    if len(codes) < n:
        for tok in out.replace(',', ' ').replace('[', ' ').replace(']', ' ').split():
            digits = ''.join(ch for ch in tok if ch.isdigit())
            if digits:
                add(digits)
            if len(codes) >= n:
                break
    return codes[:n]


def classify(text, examples, sys_p, valid_codes, prev="", nxt=""):
    """Einen Satz klassifizieren. Gibt die Rangliste zurueck oder None,
    wenn nach allen Versuchen kein gueltiger Code vorliegt."""
    messages = [{"role": "system", "content": sys_p},
                {"role": "user", "content": user_prompt(text, examples, prev, nxt)}]
    for attempt in range(MAX_RETRIES):
        try:
            resp = get_client().chat.completions.create(
                model=LLM_MODEL, messages=messages,
                temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
            content = resp.choices[0].message.content
            if content:
                ranked = parse_ranked(content.strip(), valid_codes)
                if ranked:
                    return ranked
        except Exception:
            pass
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
    return None


# ═══════════════════════════════════════════════════════════════════════
# EMBEDDINGS, POOLS UND RETRIEVAL
# ═══════════════════════════════════════════════════════════════════════

def embed_texts(texts, cache_file, chunk_size=64):
    """Embeddings berechnen; bereits vorhandene werden aus dem Cache gelesen."""
    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            cache = pickle.load(f)
    todo = [t for t in dict.fromkeys(texts) if t not in cache]
    print(f"[i] Embeddings: {len(texts) - len(todo)} aus Cache, {len(todo)} neu")
    for i in tqdm(range(0, len(todo), chunk_size), desc="Embeddings", leave=False):
        chunk = todo[i:i + chunk_size]
        for attempt in range(MAX_RETRIES):
            try:
                resp = get_client().embeddings.create(model=EMBEDDING_MODEL, input=chunk)
                for t, d in zip(chunk, resp.data):
                    cache[t] = np.array(d.embedding, dtype=np.float32)
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"[!] Embedding-Block fehlgeschlagen: {e}")
                time.sleep(RETRY_DELAY)
        if (i // chunk_size) % 20 == 0:
            atomic_pickle_dump(cache, cache_file)
    atomic_pickle_dump(cache, cache_file)
    return cache


def pool_paths(pool_year):
    return (os.path.join(POOL_DIR, f"pool_corpus_{pool_year}.json"),
            os.path.join(POOL_DIR, f"pool_emb_{pool_year}.pkl"))


def pool_from_statements(stmts, emb):
    """Pool aus Saetzen mit Goldkodierung bilden (nur eingebettete Saetze)."""
    pool, mats = [], []
    for s in stmts:
        gt = norm_code(s.get('ground_truth'))
        txt = s.get('text')
        if gt is None or not txt or txt not in emb:
            continue
        pool.append({'text': txt, 'ground_truth': gt})
        mats.append(emb[txt])
    if not pool:
        raise SystemExit("[STOP] Pool ist leer - Korpus und Embeddings pruefen.")
    return pool, np.vstack(mats).astype(np.float32)


def load_pool(pool_year):
    """Bereits aufgebauten Pool laden (wird von 02_haupttest_bundestag.py erzeugt)."""
    corpus_path, emb_path = pool_paths(pool_year)
    for p in (corpus_path, emb_path):
        if not os.path.exists(p):
            raise SystemExit(
                f"[STOP] {p} fehlt. Die Pools entstehen beim Lauf von "
                f"02_haupttest_bundestag.py (Pool 2002 mit --pool-mode era_matched).")
    with open(corpus_path, encoding='utf-8') as f:
        stmts = json.load(f)
    with open(emb_path, 'rb') as f:
        emb = pickle.load(f)
    pool, mat = pool_from_statements(stmts, emb)
    print(f"[OK] Pool {pool_year}: {len(pool)} Saetze")
    return pool, mat


def retrieve(texts, text_emb, pool, mat, k=TOP_K, batch=256):
    """Die k aehnlichsten Pool-Saetze (Kosinus-Aehnlichkeit) je Text."""
    texts = list(dict.fromkeys(texts))
    norms = np.linalg.norm(mat, axis=1)
    norms[norms == 0] = 1e-10
    out = {}
    for i in tqdm(range(0, len(texts), batch), desc="Retrieval", leave=False):
        chunk = [t for t in texts[i:i + batch] if t in text_emb]
        if not chunk:
            continue
        qs = np.vstack([text_emb[t] for t in chunk])
        qn = np.linalg.norm(qs, axis=1)
        qn[qn == 0] = 1e-10
        sims = (qs @ mat.T) / np.outer(qn, norms)
        top = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
        for j, t in enumerate(chunk):
            order = top[j][np.argsort(-sims[j, top[j]])]
            out[t] = [pool[m] for m in order]
    return out


# ═══════════════════════════════════════════════════════════════════════
# MANIFESTOBERTA
# ═══════════════════════════════════════════════════════════════════════

def context_string(prev, text, nxt):
    """Kontext wie bei der LLM-Klassifikation: ein Satz davor, einer danach."""
    return " ".join(p for p in (prev, text, nxt) if p)


def run_manifestoberta(sentences, contexts, batch_size=16):
    """Klassifikation mit ManifestoBERTa (Satz plus Kontext). Gibt je Satz
    die Rangliste der drei wahrscheinlichsten Kategorien zurueck."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    token = os.getenv("HF_TOKEN") or None
    print(f"[..] Lade {BERTA_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BERTA_MODEL, trust_remote_code=True, token=token)
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large", token=token)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.half()
    model.to(device)
    print(f"     Geraet: {device}")

    id2label = model.config.id2label
    ranked = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="ManifestoBERTa"):
        inputs = tokenizer(sentences[i:i + batch_size], contexts[i:i + batch_size],
                           return_tensors="pt", padding="max_length",
                           max_length=300, truncation=True)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        top = torch.topk(logits, k=N_RANKED, dim=1).indices.cpu().numpy()
        for row in top:
            # Labels haben die Form "201 - Freedom and Human Rights"
            codes = [norm_code(id2label[int(x)].split('-')[0].strip()) for x in row]
            ranked.append([code for code in codes if code])
    return ranked


# ═══════════════════════════════════════════════════════════════════════
# MAPPING, POSITIONEN, KENNZAHLEN
# ═══════════════════════════════════════════════════════════════════════

def load_mapping():
    """Zuordnung Kategorie -> GAL / TAN / LEFT / RIGHT / NEUTRAL."""
    df = pd.read_csv(MAPPING_CSV, sep=';', encoding='utf-8-sig', dtype=str).fillna("")
    col = [c for c in df.columns if 'GAL' in c.upper() and 'RIGHT' in c.upper()][0]
    mapping = {}
    for _, row in df.iterrows():
        code = norm_code(row['Code'])
        if code is not None:
            mapping[code] = row[col].strip().upper()
    return mapping


def position(codes, mapping, axis):
    """Normierte Differenz der beiden Pole, NEUTRAL-Saetze fliessen nicht ein.
    axis 'galtan': positiv = TAN. axis 'lr': positiv = RIGHT.
    Gibt (Position, Zahl achsenrelevanter Saetze) zurueck."""
    pos, neg = ('TAN', 'GAL') if axis == 'galtan' else ('RIGHT', 'LEFT')
    cnt = Counter(mapping.get(c) for c in codes if c)
    a, b = cnt.get(pos, 0), cnt.get(neg, 0)
    if a + b == 0:
        return float('nan'), 0
    return (a - b) / (a + b), a + b


def classification_metrics(gold, ranked_lists):
    """Top-1/2/3-Genauigkeit sowie gewichtete und Macro-Kennzahlen."""
    from sklearn.metrics import precision_recall_fscore_support
    n = len(gold)
    if n == 0:
        return {'n': 0}
    top1 = [r[0] if r else "__keine__" for r in ranked_lists]
    res = {'n': n}
    for k in (1, 2, 3):
        res[f'top{k}'] = sum(g in (r or [])[:k] for g, r in zip(gold, ranked_lists)) / n
    labels = sorted(set(gold))
    p, r, f, _ = precision_recall_fscore_support(
        gold, top1, average='weighted', zero_division=0, labels=labels)
    res.update({'precision_weighted': p, 'recall_weighted': r, 'f1_weighted': f})
    res['f1_macro'] = precision_recall_fscore_support(
        gold, top1, average='macro', zero_division=0, labels=labels)[2]
    return {k: float(v) if isinstance(v, (np.floating, float)) else v
            for k, v in res.items()}
