# Ergebnisse

Die Dateien enthalten die Ergebnisse, auf denen Kapitel 4 und 5 der Arbeit beruhen. Wahlprogrammtexte sind entfernt; die Sätze lassen sich über `manifesto_id`/`party_id`, `year` und `row_idx` den Quelldokumenten zuordnen.

Die Dateien wurden mit einer früheren Fassung der Skripte erzeugt. Klassifikationslogik, Prompt, Retrieval und Mapping sind identisch; abweichend sind nur Dateinamen und einzelne Spaltennamen (`gemma_*` statt `llm_*`).

## Haupttest (Bundestag, Kapitel 4)

| Datei | Inhalt |
|---|---|
| `haupttest/haupttest_fixed_2021.json` | Kennzahlen je Wahljahr, Rangkorrelation und MAE der Parteipositionen (LLM und ManifestoBERTa), Pool 2021 |
| `haupttest/haupttest_era_matched.json` | dasselbe mit epochenangepasstem Pool |
| `haupttest/*_codes.csv` | Kodierung je Satz: Gold, LLM-Rangliste, ManifestoBERTa-Rangliste, Achsenzuordnung |

Gewichtete Precision, Recall und F1 (Tabelle 3) lassen sich aus den `_codes.csv` mit `classification_metrics()` aus `scripts/common.py` berechnen.

## Thüringen (Kapitel 5)

| Datei | Inhalt |
|---|---|
| `thueringen/thueringen_llm_fixed_2021.json` | Positionen je Partei und Wahljahr, LLM, Pool 2021 |
| `thueringen/thueringen_llm_era_matched.json` | dasselbe mit epochenangepasstem Pool |
| `thueringen/thueringen_manifestoberta.json` | Positionen mit ManifestoBERTa und Abgleich mit dem LLM |
| `thueringen/*_codes.csv` | Kodierung je Satz |

## Daten der Abbildungen

| Datei | Abbildung / Tabelle in der Arbeit |
|---|---|
| `abbildungen/kapitel4_scatter_gold_vs_modell.csv` | Abb. 4 und 5, Tab. 9 |
| `abbildungen/kapitel4_ches2017_dreiecke.csv` | Abb. 6, Tab. 10 (CHES-Werte auf −1 bis +1 umgerechnet: (x − 5) / 5) |
| `abbildungen/kapitel5_abb_uebereinstimmung_je_jahr.csv` | Abb. 7, Tab. 11 |
| `abbildungen/kapitel5_abb_kompass_2024.csv`, `kapitel5_kompass_2024_verbindungslinien.csv` | Abb. 8, Tab. 12 |
| `abbildungen/kapitel5_abb_zeitreihe_galtan.csv` | Abb. 9, Tab. 13 |
| `abbildungen/kapitel5_abb_zeitreihe_links_rechts.csv` | Abb. 10, Tab. 14 |
| `abbildungen/kapitel5_abb_trajektorie.csv` | Abb. 11 |
| `abbildungen/kapitel5_anhang_positionen_thueringen.csv` | alle Positionen je Partei und Wahljahr in beiden Pool-Modi und mit ManifestoBERTa |

Die CSV-Dateien im Ordner `abbildungen` verwenden das deutsche Format (Semikolon als Trennzeichen, Komma als Dezimalzeichen).
