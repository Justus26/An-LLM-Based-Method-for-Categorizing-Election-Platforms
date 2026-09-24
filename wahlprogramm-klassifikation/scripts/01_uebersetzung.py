"""
Schritt 1: Uebersetzung der segmentierten Wahlprogramme ins Englische.

Die Klassifikation arbeitet auf Englisch, weil Beispiel-Pool und Modelle
englischsprachig sind. Uebersetzt wird mit einem Chat-Modell ueber eine
OpenAI-kompatible Schnittstelle.

Absicherungen
  - Saetze werden nummeriert in Batches geschickt. Die Antwort muss exakt
    so viele Elemente enthalten wie gesendet wurden; sonst wird der Batch
    halbiert und erneut versucht (bis hinunter zu Einzelsaetzen). So bleibt
    jede Uebersetzung dem richtigen Satz zugeordnet und die Segmentgrenzen
    bleiben erhalten.
  - Uebersetzungen mit fremden Schriftzeichen (z.B. chinesische Zeichen,
    die mehrsprachige Modelle gelegentlich einstreuen) werden einzeln neu
    uebersetzt oder als fehlend markiert.

Eingabe  data/thueringen/segmente_de.json
         Liste von Saetzen mit den Feldern manifesto_id, year, party,
         party_id, row_idx, text_de, prev_de, next_de
Ausgabe  data/thueringen/segmente_en.csv

Aufruf   python scripts/01_uebersetzung.py
"""

import json
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA_DIR, MAX_RETRIES, MAX_WORKERS, get_client, atomic_write_json

TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "qwen3-next-80b-a3b-instruct")

IN_JSON = os.path.join(DATA_DIR, "thueringen", "segmente_de.json")
OUT_CSV = os.path.join(DATA_DIR, "thueringen", "segmente_en.csv")
CHECKPOINT = os.path.join(DATA_DIR, "thueringen", "uebersetzung_checkpoint.json")

BATCH_SIZE = 15
RETRY_DELAY = 3
CHECKPOINT_EVERY = 200

SYSTEM_PROMPT = """You are translating sentences from German political party \
manifestos into British English, in the register the Manifesto Project uses \
for its own official translations: formal, literal, policy-register English \
- not a loose paraphrase.

You will receive a numbered list of German sentences. Translate EACH one \
independently. Do not merge sentences, do not split a sentence into two, do \
not add commentary, do not change the order.

Respond with EXCLUSIVELY a JSON array of strings, in the same order, with \
EXACTLY as many elements as input sentences. No other text."""

ALLOWED_NON_ASCII = "äöüÄÖÜßéèêáàâíìîóòôúùûñç–—‘’“”…°%€$§"


def has_non_latin_leak(text):
    """True, wenn die Uebersetzung Buchstaben oder Ziffern aus fremden
    Schriftsystemen enthaelt."""
    for ch in text:
        if ch.isascii() or ch in ALLOWED_NON_ASCII:
            continue
        if unicodedata.category(ch).startswith(('L', 'N')):
            return True
    return False


def call_llm(batch):
    lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(batch))
    prompt = f"{lines}\n\nJSON array of {len(batch)} translations:"
    for attempt in range(MAX_RETRIES):
        try:
            resp = get_client().chat.completions.create(
                model=TRANSLATION_MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=max(200, 80 * len(batch)))
            content = resp.choices[0].message.content.strip()
            content = re.sub(r'^```(json)?|```$', '', content, flags=re.MULTILINE).strip()
            arr = json.loads(content[content.index('['):content.rindex(']') + 1])
            if isinstance(arr, list):
                return [str(x) for x in arr]
        except Exception:
            pass
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY)
    return None


def translate_batch_safe(batch):
    """Batch uebersetzen; bei falscher Elementzahl halbieren (Bisektion)."""
    if not batch:
        return {}
    result = call_llm(batch)
    if result is None or len(result) != len(batch):
        if len(batch) == 1:
            return {}
        mid = len(batch) // 2
        out = translate_batch_safe(batch[:mid])
        out.update(translate_batch_safe(batch[mid:]))
        return out

    out = dict(zip(batch, result))
    for src in [s for s, en in out.items() if has_non_latin_leak(en)]:
        retry = call_llm([src])
        if retry and len(retry) == 1 and not has_non_latin_leak(retry[0]):
            out[src] = retry[0]
        else:
            del out[src]
    return out


def main():
    if not os.path.exists(IN_JSON):
        raise SystemExit(f"{IN_JSON} fehlt - erst die Segmentierung ausfuehren.")
    with open(IN_JSON, encoding='utf-8') as f:
        rows = json.load(f)

    unique = list(dict.fromkeys(r['text_de'] for r in rows))
    done = {}
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, encoding='utf-8') as f:
            done = json.load(f)
    todo = [t for t in unique if t not in done]
    print(f"[i] {len(rows)} Segmente, {len(unique)} verschiedene Saetze, "
          f"{len(todo)} offen (Modell: {TRANSLATION_MODEL})")

    batches = [todo[i:i + BATCH_SIZE] for i in range(0, len(todo), BATCH_SIZE)]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [ex.submit(translate_batch_safe, b) for b in batches]
        for i, fut in enumerate(tqdm(futures, desc="Uebersetzen")):
            done.update(fut.result())
            if (i + 1) % CHECKPOINT_EVERY == 0:
                atomic_write_json(done, CHECKPOINT)
    atomic_write_json(done, CHECKPOINT)

    missing = sum(1 for t in unique if t not in done)
    if missing:
        print(f"[!] {missing} Saetze ohne Uebersetzung - Skript erneut starten")

    for r in rows:
        r['text_en'] = done.get(r['text_de'], '')
        r['prev_en'] = done.get(r.get('prev_de', ''), '')
        r['next_en'] = done.get(r.get('next_de', ''), '')

    cols = ['manifesto_id', 'year', 'party', 'party_id', 'row_idx',
            'text_de', 'text_en', 'prev_de', 'prev_en', 'next_de', 'next_en']
    df = pd.DataFrame(rows)
    df[[c for c in cols if c in df.columns]].to_csv(OUT_CSV, index=False, encoding='utf-8')
    print(f"[OK] {OUT_CSV}: {len(df)} Segmente")


if __name__ == "__main__":
    main()
