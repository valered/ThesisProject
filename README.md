# Commit Message Generation — Comparative Study

Questa repository contiene il codice, gli script di valutazione, e i risultati sperimentali principali utilizzati nella tesi magistrale dedicata alla generazione automatica di messaggi di commit.

Lo studio confronta tre modelli rappresentativi:
- **CoDiSum** (2019): RNN con meccanismo di copia e input strutturale.
- **RACE** (2022): Transformer guidato dal retrieval di commit simili.
- **KADEL** (2024): Approccio knowledge-aware con denoising semantico e template-guided learning.

---

## 📦 Contenuto della repository

- `CoDiSum/`, `RACE/`, `KADEL/`: script principali, moduli neurali, configurazioni e risultati leggeri per ciascun modello.
- `plots_metrics/`: visualizzazioni comparative delle metriche.
- `stats_race_vs_kadel.csv`: riepilogo dei risultati su D2.
- `Thesis.pdf`: testo completo della tesi.

---

## 📁 Dataset

I dataset completi non sono inclusi nella repository per motivi di dimensioni. Sono comunque disponibili al seguente link pubblico:

🔗 **[Google Drive - dataset_tesi](https://drive.google.com/drive/folders/1r5zjBBs8eNGnGOdK2rM2gMIHzwzwmOE9?usp=drive_link)**

La struttura della cartella è:

dataset_tesi/
├── RACE/
│ └── file .jsonl con retrieval (test/train/valid)
├── KADEL/
│ └── dataset jsonl compatibili con la pipeline di addestramento



> I file sono stati esclusi per via del superamento dei limiti di GitHub (>100MB). Tutti gli script e i percorsi sono già pronti per lavorare con questi dati una volta scaricati e posizionati nella struttura corretta.

---

## ❌ Componenti escluse dalla repository

Per motivi di spazio, sicurezza o duplicazione, non sono stati inclusi:

- I modelli preaddestrati (es. `*.h5`, `*.bin`, `*.ckpt`, `.pyd`, `.dll`)
- Ambienti virtuali o cartelle locali (`venv/`, `codisum-env/`)
- File intermedi o di log (`*.log`, `*.pickle`, `__pycache__`)
- Cartelle di output pesanti (es. `saved_model/`, `trunc/`, `cache/`)
- Tutti i file già disponibili nelle repository ufficiali (vedi sotto)

---

## 📚 Repository ufficiali di riferimento

I modelli originali sono disponibili nelle seguenti repository ufficiali:

- 🔗 CoDiSum (Xu et al., 2019):  
  https://github.com/SoftWiser-group/CoDiSum

- 🔗 RACE (Shi et al., 2022):  
  https://github.com/DeepSoftwareAnalytics/RACE

- 🔗 KADEL (Tao et al., 2024):  
  https://github.com/DeepSoftwareAnalytics/KADEL

Nella presente repository sono stati utilizzati solo gli script, i file e le risorse effettivamente impiegati nel lavoro di tesi, rimuovendo componenti non usate o ridondanti.

---

## ⚙️ Setup

Il progetto è stato eseguito su **Google Colab Pro** con:
- Python ≥ 3.8
- TensorFlow 2.20.0 / PyTorch 2.3.0
- Transformers (v4.x), NLTK, SacreBLEU, ROUGE, METEOR
- TensorBoard, tqdm, scikit-learn

---

## 📊 Valutazione

Metriche usate:
- BLEU
- ROUGE-1/2/L
- METEOR
- SacreBLEU
- CodeBLEU-text
- CIDEr
- Identifier Recall

Valutazione basata su più run indipendenti per ciascun modello, con analisi statistica di stabilità, significatività e riproducibilità.

---

## 📄 Tesi

Il testo completo della tesi è disponibile nel file `Thesis.pdf` incluso nella root della repository.
