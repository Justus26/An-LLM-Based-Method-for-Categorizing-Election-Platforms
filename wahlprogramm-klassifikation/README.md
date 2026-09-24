# Analyse von Wahlprogrammen mit LLM-Klassifikation und GAL-TAN-Mapping

Code zur Bachelorarbeit *„Analyse von Wahlprogrammen mithilfe computergestützter Methoden: Politische Kategorisierung durch LLM und GAL-TAN-Visualisierung am Beispiel Thüringer Landtagswahlen 1990–2024"* (Universität Passau, 2026).

Die Quasi-Sätze von Wahlprogrammen werden mit einem großen Sprachmodell den 56 Kategorien des Manifesto Project zugeordnet (Few-Shot-Klassifikation mit dynamisch abgerufenen, handkodierten Beispielsätzen). Anschließend werden die Kategorien auf eine gesellschaftliche Achse (GAL-TAN) und eine ökonomische Achse (Links-Rechts) abgebildet und daraus Parteipositionen berechnet.

## Ablauf

| Skript | Zweck |
|---|---|
| `scripts/00_segmentierung.py` | Zerlegung der Thüringer Programme in Quasi-Sätze |
| `scripts/00b_segmentierung_validierung.py` | Abgleich der Segmentierung mit der Handsegmentierung des Manifesto Project (FDP 2025) |
| `scripts/01_uebersetzung.py` | Übersetzung der segmentierten Thüringer Programme ins Englische |
| `scripts/02_haupttest_bundestag.py` | Modellvergleich (LLM, ManifestoBERTa) gegen die Handkodierung auf Bundestagsprogrammen 1998–2025; baut dabei die Beispiel-Pools auf |
| `scripts/03_thueringen_llm.py` | Klassifikation der Thüringer Programme und Positionsberechnung |
| `scripts/04_thueringen_manifestoberta.py` | ManifestoBERTa als zweiter, unabhängiger Klassifikator für Thüringen |
| `scripts/common.py` | gemeinsame Funktionen (Prompt, Retrieval, Mapping, Kennzahlen) |
| `scripts/segmentation.py` | Segmentierungslogik |

Reihenfolge: 02 (mit beiden Pool-Modi) → 00 → 01 → 03 → 04. Die Validierung 00b ist unabhängig davon.

## Einrichtung

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_lg
cp .env.example .env      # danach Zugangsdaten eintragen
```

Benötigt werden eine OpenAI-kompatible Schnittstelle für Sprach- und Embedding-Modell sowie ein kostenloser API-Schlüssel des [Manifesto Project](https://manifesto-project.wzb.eu/). ManifestoBERTa läuft lokal und profitiert deutlich von einer GPU.

### Konfigurationsdateien (`config/`)

- `gal_tan_left_right_mapping.csv`: Zuordnung aller Kategorien zu GAL, TAN, LEFT, RIGHT oder NEUTRAL mit Begründung
- `categories.csv`: Kategorienliste für den Prompt (Spalten `Code;Name`)

### Daten (`data/`, nicht im Repository)

Wahlprogramme und daraus abgeleitete Dateien sind nicht enthalten, da sie den Nutzungsbedingungen der jeweiligen Quellen unterliegen.

- **Bundestagswahlprogramme:** werden über die Manifesto-API abgerufen. Noch nicht über die API verfügbare Jahrgänge als `data/manifestos_local/<party_id>_<YYYYMM>.csv` ablegen (Spalten `text`, `text_en`, `cmp_code`).
- **Thüringer Landtagswahlprogramme:** Download über [PolDoc](https://polidoc.net/), Textdateien als `data/poldoc/<manifesto_id>.txt` ablegen (z. B. `41223.016.2019.1.1.txt`).
- **Validierung der Segmentierung:** Rohtext des FDP-Programms 2025 als `data/validierung/41420.000.2025.1.1.txt`, Handsegmentierung als `data/manifestos_local/41420_202502.csv`.
- **Beispiel-Pools:** entstehen in `data/pools/` beim Lauf von Skript 02.

## Aufruf

```bash
python scripts/00b_segmentierung_validierung.py
python scripts/02_haupttest_bundestag.py --pool-mode fixed_2021
python scripts/02_haupttest_bundestag.py --pool-mode era_matched --no-berta
python scripts/00_segmentierung.py
python scripts/01_uebersetzung.py
python scripts/03_thueringen_llm.py --pool-mode fixed_2021
python scripts/03_thueringen_llm.py --pool-mode era_matched
python scripts/04_thueringen_manifestoberta.py --compare results/thueringen/thueringen_llm_fixed_2021.csv
```

Alle Skripte speichern Zwischenstände und können nach einem Abbruch neu gestartet werden, ohne bereits klassifizierte Sätze erneut zu senden.

## Methodische Eckdaten

| Parameter | Wert |
|---|---|
| Klassifikationsmodell | gemma4-31b-it |
| Embedding-Modell | octen-embedding-8b |
| Übersetzungsmodell | Qwen3-Next-80B-A3B-Instruct |
| Referenzmodell | ManifestoBERTa (56policy-topics-context-2026-1-1) |
| Beispiele je Satz | 15 (Kosinus-Ähnlichkeit) |
| Beispiel-Pool | 5.900 handkodierte Sätze (Bundestag 2021 bzw. 2002) |
| Temperatur | 0,1 |
| Position | (TAN − GAL) / (TAN + GAL) bzw. (RIGHT − LEFT) / (RIGHT + LEFT); neutrale Sätze fließen nicht ein |
| Mindestfallzahl | 30 achsenrelevante Sätze je Partei und Wahl |

Bei einer Temperatur von 0,1 sind die Ergebnisse nicht exakt deterministisch; zwei Läufe unterscheiden sich in rund 0,5 % der Kategorien.

## Hinweis zur Entstehung

Der Code wurde unter Mitwirkung von KI-Assistenz entwickelt und durch den Autor geprüft (siehe Kapitel 3.1 der Arbeit).

## Lizenz

Code: MIT-Lizenz (siehe `LICENSE`).
