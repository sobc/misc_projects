# tug_minimal

Dieses Subprojekt enthält zwei Varianten desselben Diffusions-Benchmarks:

- `tug-old` (alte Implementierung)
- `tug-optimized` (optimierte Implementierung)

Beide Programme lesen Benchmark-Eingaben aus dem Projekt, führen eine gegebene
Anzahl Iterationen aus und schreiben das Ergebnis als CSV.

## Voraussetzungen

- CMake >= 3.24
- C++17-fähiger Compiler (z. B. GCC oder Clang)
- OpenMP-Unterstützung im Compiler
- Internetzugang beim ersten Konfigurieren (Abhängigkeiten werden über CPM geladen: Eigen, CLI11)

## Kompilieren

Im Verzeichnis `tug_minimal`:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Optional für Profiling/Debugging mit Symbolen:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
```

## Ausführen

Allgemeine Syntax:

```bash
./build/tug-old [OPTIONS] iterations
./build/tug-optimized [OPTIONS] iterations
```

- `iterations` ist ein Pflichtparameter (Anzahl Simulationsschritte)
- Option `-o, --output` setzt den Pfad zur Ausgabe-CSV

### Beispiele

Ein kurzer Testlauf mit 1 Iteration, ohne dass eine Ausgabe-CSV erstellt wird:

```bash
./build/tug-old 1 -o /dev/null
./build/tug-optimized 1 -o /dev/null
```

Lauf mit 100 Iterationen, wobei die Ausgabe-CSV automatisch benannt wird:

```bash
./build/tug-old 100 
./build/tug-optimized 100 ```

## Hilfe anzeigen

```bash
./build/tug-old --help
./build/tug-optimized --help
```

## Hinweise

- Ohne `-o/--output` wird automatisch ein Dateiname auf Basis des Benchmarks verwendet.
- Beide Targets verwenden dieselben Eingabedaten (`barite_large`) und
  unterscheiden sich nur in der Implementierung (`old/` vs. `optimized/`). 
- Die Eingabedaten befinden sich im Verzeichnis `tug_minimal/inputs/` und dürfen
  nicht verschoben oder gelöscht werden.
