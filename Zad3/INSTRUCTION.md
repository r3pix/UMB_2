# Projekt 2 – Uczenie maszynowe w bezpieczeństwie (Zadania 1–3)

## Co jest w repozytorium
- `src/task1_task2.py` – kompletna implementacja Zadania 1 i 2 (Alg. 1–4), generuje metryki i wykresy do `results/`
- `src/task3_real_data.py` – kompletna implementacja feature engineering dla CICIDS2017 i UNSW-NB15 + trening/evaluacja (60/20/20, LR L2, C=1.0, class_weight=balanced)
- `results/` – wyniki wygenerowane lokalnie (po uruchomieniu `python run_all.py`)
- `requirements.txt` – zależności

## Jak uruchomić
```bash
pip install -r requirements.txt
python run_all.py
python run_task3.py
python run_task3_plots.py
```

## Zadanie 3 – dane
Włóż pliki do `data/` i uruchom kod z `src/task3_real_data.py` po uzupełnieniu ścieżek w sekcji `__main__` lub wywołaj funkcje `load_cicids(...)` / `load_unsw(...)`.

CICIDS2017: użyj Monday-WorkingHours (BENIGN) + Friday-WorkingHours-Afternoon-DDoS + Friday-WorkingHours-Afternoon-PortScan.
