"""
Segmentierung von Wahlprogrammen in Quasi-Saetze (Naeherung an die
Manifesto-Konvention).

Vorgehen
  1. Aufzaehlungszeilen ("- ...") gelten als eigene Einheiten (eine
     Forderung je Aufzaehlungspunkt) und laufen nicht durch den
     Satzsplitter, der auf kurzen Fragmenten unzuverlaessig ist.
  2. Fliesstext wird mit spaCy in Saetze zerlegt.
  3. Zusaetzlich wird an Doppelpunkt und Semikolon geschnitten, sofern
     beide Teile mindestens MIN_LEN_AFTER_SPLIT Woerter haben. Ein Schnitt
     an Konjunktionen ("und", "sowie") hat sich als zu grob erwiesen, weil
     sie meist innerhalb eines Quasi-Satzes stehen.
  4. Kein Text wird verworfen: sehr kurze Reste werden an das vorherige
     Segment angehaengt. Die Reihenfolge des Dokuments bleibt erhalten,
     damit der Kontext (vorheriger/naechster Satz) stimmt.

Benoetigt: python -m spacy download de_core_news_lg
"""

import re

NLP_MODEL = "de_core_news_lg"
MIN_LEN_AFTER_SPLIT = 4   # Woerter; kuerzere Teile werden nicht abgetrennt
MIN_SEGMENT_WORDS = 3     # kuerzere Segmente werden an das vorherige angehaengt


def load_nlp():
    import spacy
    return spacy.load(NLP_MODEL)


def split_punctuation(sent):
    """An Doppelpunkt und Semikolon schneiden, wenn beide Teile lang genug sind."""
    parts = re.split(r'(?<=[:;])\s+', sent)
    out, buf = [], ""
    for p in parts:
        buf = (buf + " " + p).strip() if buf else p
        if len(buf.split()) >= MIN_LEN_AFTER_SPLIT:
            out.append(buf)
            buf = ""
    if buf:
        if out and len(buf.split()) < MIN_LEN_AFTER_SPLIT:
            out[-1] = out[-1] + " " + buf
        else:
            out.append(buf)
    return out if out else [sent]


def segment(raw, nlp):
    """Rohtext eines Wahlprogramms in Quasi-Saetze zerlegen."""
    blocks, buf = [], []

    def flush_prose():
        if buf:
            blocks.append(('prose', " ".join(buf)))
            buf.clear()

    for line in raw.split("\n"):
        s = line.strip()
        if s.startswith("- ") or s == "-":
            flush_prose()
            content = s.lstrip("- ").strip()
            if content:
                blocks.append(('bullet', content))
        elif s == "":
            flush_prose()
        else:
            buf.append(s)
    flush_prose()

    segments = []
    for kind, block in blocks:
        if kind == 'bullet':
            segments.extend(split_punctuation(block))
        else:
            for sent in nlp(block).sents:
                segments.extend(split_punctuation(sent.text.strip()))

    cleaned = []
    for s in (s.strip() for s in segments):
        if not s:
            continue
        if len(s.split()) < MIN_SEGMENT_WORDS and cleaned:
            cleaned[-1] = cleaned[-1].rstrip('.') + " " + s
        else:
            cleaned.append(s)
    return cleaned
