# BERT Multi-Label Klassifikation (Gesundheitstexte)

## Übersicht

Dieses Repository enthält die Trainingspipeline für ein **transformerbasiertes Klassifikationsmodell** zur Analyse kurzer, deutschsprachiger Freitexte im Gesundheitskontext.

Das Modell basiert auf **`deepset/gbert-base` (BERT für Deutsch)** und wird für eine **Multi-Label-Klassifikation** der folgenden Kategorien trainiert:

* Stress
* Schlaf
* Bewegung
* Ernährung

Ziel ist es, kurze Texte automatisch mehreren relevanten Problembereichen zuzuordnen.

---

## Funktionsweise

Die Pipeline umfasst folgende Schritte:

1. Laden eines CSV-Datensatzes (`data.csv`)
2. Train/Test-Split (80/20)
3. Tokenisierung mit BERT-Tokenizer
4. Training eines Multi-Label-Klassifikationsmodells
5. Evaluation mit:

   * Micro / Macro F1
   * Precision / Recall
6. Speicherung von:

   * Modell
   * Vorhersagen
   * Fehleranalyse

---

## Datensatz

Der Datensatz muss folgende Struktur haben:

| text                      | stress | schlaf | bewegung | ernaehrung |
| ------------------------- | ------ | ------ | -------- | ---------- |
| "Ich schlafe schlecht..." | 1      | 1      | 0        | 0          |

* `text`: Freitext (String)
* Labels: 0 oder 1 (Multi-Label möglich)

---

## Installation

Benötigte Libraries (Auszug):

* transformers
* datasets
* torch
* scikit-learn
* pandas

---

## Training starten

```bash
python train.py
```

---

## Output

Alle Ergebnisse werden im Ordner `gbert_multilabel_output/` gespeichert:

* `best_model/` → trainiertes Modell + Tokenizer
* `test_predictions.csv` → Vorhersagen auf Testdaten
* `error_analysis.csv` → Fehlklassifikationen
* `manual_examples_predictions.csv` → Beispielvorhersagen

---

## Modell

* Basis: `deepset/gbert-base`
* Task: Multi-Label Classification
* Loss: automatisch durch HuggingFace (`BCEWithLogitsLoss`)
* Threshold: 0.5 (konfigurierbar)

---

## Besonderheiten

* Multi-Label statt Single-Label Klassifikation
* Explizite Fehleranalyse pro Beispiel
* Evaluation mit Micro und Macro Metrics
* Geeignet für kurze, alltagsnahe Texte

---

## Beispiel

Input:

```
ich esse und schlafe zu wenig
```

Output (Beispiel):

```
Schlaf, Ernährung
```

---

## Hinweis

Das Modell wurde im Rahmen einer **prototypischen Anwendung** entwickelt.
Die Leistung hängt stark von der Qualität und Struktur des Trainingsdatensatzes ab.
