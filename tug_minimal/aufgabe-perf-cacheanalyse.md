# Aufgabe: Cacheanalyse mit `perf`

## Kurzbeschreibung des Programms

Dieses Projekt vergleicht zwei Versionen desselben numerischen Diffusionslösers: eine alte Variante (`tug-old`) und eine optimierte Variante (`tug-optimized`). Das Programm liest Eingabedaten aus CSV-Dateien, baut daraus ein Gitter mit Randbedingungen auf, führt mehrere Iterationen einer 2D-Diffusionssimulation aus und schreibt das Ergebnis wieder als CSV-Datei.

## Ziel der Aufgabe

Analysieren Sie mit Hilfe von `perf`, wie sich das Speicher- und Cacheverhalten der alten und der optimierten Version unterscheidet. Nutzen Sie diese Messungen anschließend, um den Leistungsunterschied mit eigenen Worten zu erklären.

## Vorbereitung

1. Bauen Sie beide Programme:

   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
   cmake --build build -j
   ```

2. Führen Sie beide Programme einmal ohne `perf` aus, damit Sie sehen, wie sie gestartet werden:

   ```bash
   ./build/tug-old 1 -o /tmp/tug_old.csv
   ./build/tug-optimized 1 -o /tmp/tug_optimized.csv
   ```

## Aufgaben

### 1. Cachezugriffe messen

Messen Sie für beide Programme die Cachezugriffe und Cache-Fehlzugriffe mit `perf`.

Verwenden Sie zum Beispiel:

```bash
perf stat -r 3 -e cache-references,cache-misses,cycles,instructions ./build/tug-old 1 -o /tmp/tug_old.csv
perf stat -r 3 -e cache-references,cache-misses,cycles,instructions ./build/tug-optimized 1 -o /tmp/tug_optimized.csv
```

Falls auf Ihrem System weitere Cache-Metriken verfügbar sind, dürfen Sie diese zusätzlich verwenden.

### 2. Ergebnisse vergleichen

Vergleichen Sie mindestens die folgenden Punkte zwischen `tug-old` und `tug-optimized`:

- Anzahl der `cache-references`
- Anzahl der `cache-misses`
- Verhältnis von Cache-Misses zu Cache-References
- Laufzeit bzw. die von `perf` gemessenen Gesamtwerte

Stellen Sie Ihre Ergebnisse übersichtlich dar, zum Beispiel in einer kleinen Tabelle.

### 3. Unterschied fachlich erklären

Erklären Sie mit Ihren eigenen Worten, warum sich die alte und die optimierte Version unterscheiden. Beziehen Sie sich dabei ausdrücklich auf Ihre `perf`-Messungen.

Gehen Sie dabei auf folgende Fragen ein:

1. Welche Version verhält sich in Bezug auf die Caches günstiger?
2. Woran erkennen Sie das in den `perf`-Ergebnissen?
3. Wie lässt sich damit der Geschwindigkeitsunterschied erklären?
4. Welche Rolle spielt dabei die optimierte Speicherzugriffsstruktur im Code?

## Abgabe

Geben Sie eine kurze Auswertung ab, die aus drei Teilen besteht:

1. Einer sehr kurzen Beschreibung der Funktion des Programms
2. Den `perf`-Messwerten für beide Programme
3. Ihrer eigenen Erklärung des Unterschieds zwischen alter und optimierter Version auf Basis der Messungen

## Hinweise

- Verwenden Sie für beide Programme dieselben Parameter, damit die Ergebnisse vergleichbar bleiben.
- Wenn einzelne Messwerte leicht schwanken, orientieren Sie sich an den Trends über mehrere Läufe.
- Konzentrieren Sie sich in Ihrer Erklärung auf Cacheverhalten und Speicherzugriffe, nicht nur auf die reine Laufzeit.
