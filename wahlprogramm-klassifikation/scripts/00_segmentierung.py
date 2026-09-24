"""
Schritt 0: Segmentierung der Thueringer Landtagswahlprogramme.

Zerlegt die Programme in Quasi-Saetze (Logik in segmentation.py) und
schreibt sie mit Metadaten und Kontext (vorheriger und naechster Satz
desselben Programms) in eine Datei, die 01_uebersetzung.py einliest.

Eingabe  data/poldoc/<manifesto_id>.txt, z.B. 41223.016.2019.1.1.txt
         Schema der PolDoc-Kennung: <Partei-ID>.<Region>.<Jahr>.<Version>.<Version>
         (Region 016 = Thueringen)
Ausgabe  data/thueringen/segmente_de.json

Aufruf   python scripts/00_segmentierung.py
"""

import glob
import os
import sys

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA_DIR, atomic_write_json
from segmentation import load_nlp, segment

INPUT_DIR = os.path.join(DATA_DIR, "poldoc")
OUTPUT_JSON = os.path.join(DATA_DIR, "thueringen", "segmente_de.json")

# Parteien ohne Bezug zur Fragestellung (Piratenpartei)
EXCLUDE_PARTY_IDS = {"41950", "41952"}

PARTY_NAMES = {
    '41112': 'Bündnis 90', '41113': 'Grüne',
    '41221': 'PDS', '41223': 'Die Linke',
    '41320': 'SPD', '41420': 'FDP', '41440': 'FW',
    '41521': 'CDU', '41702': 'NPD', '41953': 'AfD',
}


def parse_id(manifesto_id):
    parts = manifesto_id.split('.')
    return {'party_id': parts[0], 'region': parts[1], 'year': parts[2]}


def main():
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.txt")))
    if not files:
        raise SystemExit(f"Keine Textdateien in {INPUT_DIR} gefunden.")
    nlp = load_nlp()

    rows, skipped = [], []
    for path in tqdm(files, desc="Segmentierung"):
        manifesto_id = os.path.splitext(os.path.basename(path))[0]
        meta = parse_id(manifesto_id)
        if meta['party_id'] in EXCLUDE_PARTY_IDS:
            skipped.append(manifesto_id)
            continue
        with open(path, encoding='utf-8', errors='ignore') as f:
            segments = segment(f.read(), nlp)
        for i, text in enumerate(segments):
            rows.append({
                'manifesto_id': manifesto_id,
                'year': meta['year'],
                'party': PARTY_NAMES.get(meta['party_id'], meta['party_id']),
                'party_id': meta['party_id'],
                'region': meta['region'],
                'row_idx': i,
                'text_de': text,
                'prev_de': segments[i - 1] if i > 0 else "",
                'next_de': segments[i + 1] if i < len(segments) - 1 else "",
            })

    if skipped:
        print(f"[i] ausgeschlossen: {', '.join(skipped)}")
    unknown = sorted({r['party_id'] for r in rows} - set(PARTY_NAMES))
    if unknown:
        print(f"[!] Partei-IDs ohne Namen in PARTY_NAMES: {', '.join(unknown)}")
    atomic_write_json(rows, OUTPUT_JSON)
    print(f"[OK] {len(rows)} Segmente aus {len(files) - len(skipped)} Programmen "
          f"-> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
