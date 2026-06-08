# CHARTA — AI Agent Implementation Guide
### Clinical History-Aware Temporal Architecture
> **Version:** 6.0 | **Target:** MSc Final Year Project | **Optimised for:** AI-assisted development
> **Changelog v6.0:** Resolved 6 new bugs (BUG-N1 through BUG-N6) identified by code review: fixed undefined `linker` variable in `entity_linker.py`; fixed `entity_name` vs `concept_id` confusion in `feature_explainer.py`; fixed `KeyError` on `None` concept_id in `graph_builder.py`; fixed double-sigmoid training instability in `readmission_head.py` + `trainer.py`; clarified end-to-end pipeline data path; fixed missing `logs/` directory guard in `utils.py`. Fixed RISK-2 (datasets version pin corrected to 2.21.0). Fixed RISK-4 (missing steps 21–22 renumbered). Fixed `.gitignore` to preserve `adapter_config.json`. Updated Architecture Rules to reflect .pt inter-layer communication. Fixed OpenI patient UID naming for multi-visit graph grouping. Added `name_index` to `graph_meta.json` for human-readable entity names in Layer 5 reports.
>
> **Changelog v5.0:** Simplified Layer 1 to plain text (.txt) preprocessing only — removed PDF, OCR, and image ingestion; simplified Layer 4 to single-task readmission risk prediction — removed FAISS, RAG, and multi-head inference; simplified Layer 5 to template-based explainable report generation — removed BioGPT, LoRA fine-tuning, and SHAP complexity.

> **How to use this guide:** Each step is self-contained and has a ✅ VERIFY checkpoint.
> An AI agent should execute one step, confirm the checkpoint passes, then move to the next.
> Never skip a checkpoint — every layer depends on the one before it.

---

## Table of Contents

0. [Prerequisites](#0-prerequisites)
1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [Technology Stack & requirements.txt](#3-technology-stack--requirementstxt)
4. [Dataset Overview & Access Verification](#4-dataset-overview--access-verification)
5. [Layer-wise Breakdown](#5-layer-wise-breakdown)
6. [Execution Plan — Step by Step](#6-execution-plan--step-by-step)
7. [Constraints and Rules](#7-constraints-and-rules)
8. [Testing Strategy](#8-testing-strategy)
9. [Evaluation Metrics](#9-evaluation-metrics)
10. [Known Bug Registry](#10-known-bug-registry)

---

## 0. Prerequisites

> **AI Agent instruction:** Read this entire section before executing any step.
> Confirm every prerequisite is met. If any item fails, resolve it before proceeding.

---

### 0.1 Hardware Requirements

| Resource | Minimum | Recommended | Notes |
|---|---|---|---|
| RAM | 8 GB | 16 GB | ClinicalBERT loads ~1.3 GB; OpenI images add ~2 GB |
| Disk | 20 GB free | 40 GB free | Datasets + models + intermediate files |
| CPU | 4 cores | 8 cores | Layers 1–3 run on CPU only |
| GPU | Not required locally | — | Training (Layer 4 Phase 2 + Layer 5 Phase 3) runs on Google Colab free T4 |
| Internet | Required | Stable | ~3 GB of datasets and models to download |

---

### 0.2 Software Prerequisites

**Operating System:** Windows 10/11 (64-bit). All commands in this guide are Windows CMD unless marked `[Colab]`.

**Python 3.10 — exact version required.**

```cmd
REM Check your Python version first
python --version
REM Must print: Python 3.10.x
REM If not, download from: https://www.python.org/downloads/release/python-31011/
REM During install: CHECK "Add Python 3.10 to PATH" ← critical
```

**Git:**
```cmd
git --version
REM If missing: https://git-scm.com/download/win
```

**Google Colab access (for GPU training steps only):**
- Free Google account at `colab.research.google.com`
- No subscription needed — T4 GPU is available on the free tier
- Training steps that require Colab are clearly marked `[Colab]`

---

### 0.3 Knowledge Prerequisites

This guide assumes you can:
- Run commands in Windows CMD / PowerShell
- Edit Python files in VS Code or any text editor
- Understand basic Python classes and functions
- Follow numbered steps in order

This guide does NOT assume:
- Prior knowledge of GNNs, Transformers, or graph databases
- Experience with clinical NLP
- GPU programming knowledge

---

### 0.4 Accounts and Access

| Service | Required? | Register at | Time to activate |
|---|---|---|---|
| GitHub | No (for dataset download) | — | — |
| HuggingFace | No (all models/datasets load anonymously) | — | — |
| Kaggle | No (using GitHub mirror) | — | — |
| Google (Colab) | Yes, for GPU training | accounts.google.com | Instant |
| NIH/UMLS | **No** — MedCAT removed, replaced by scispaCy EntityLinker | — | — |
| PhysioNet/MIMIC | **No** — MIMIC removed from this project | — | — |

**Zero gated datasets are used in CHARTA v3.0.**

---

### 0.5 Pre-flight Check

Run this before starting Step 1. All three lines must succeed:

```cmd
python --version
REM → Python 3.10.x

git --version
REM → git version 2.x.x

python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=5); print('Internet OK')"
REM → Internet OK
```

If all three pass: proceed to Step 1.

---

## 1. Project Overview

CHARTA is an end-to-end clinical AI system that:

- **Preprocesses** plain text clinical notes (cleaning, normalisation, sentence segmentation)
- **Extracts** clinical entities (diagnoses, drugs, lab values) using biomedical NLP
- **Builds** a patient-level temporal knowledge graph across multiple visits
- **Predicts** 30-day readmission risk using a Graph Neural Network over the patient history graph
- **Explains** every risk score in plain English using feature importance and template-based report generation

**Input:** Folder of plain text (.txt) clinical notes
**Output:** Readmission risk score + readable clinical explanation

### Architecture in one line

```
Clinical text → [L1 Preprocess] → [L2 NLP] → [L3 Graph] → [L4 Risk] → [L5 Explain] → Patient report
```

---

## 2. Folder Structure

```
CHARTA/
│
├── .gitignore
├── conftest.py                        # Pytest sys.path config
├── requirements.txt
├── run_layer1.py
├── run_layer2.py
├── run_layer3.py
├── run_layer4.py
├── run_layer5.py
├── run_pipeline.py
│
├── data/
│   ├── raw/
│   │   └── txt/
│   ├── processed/                     # Layer 1 output
│   ├── extracted/                     # Layer 2 output
│   ├── graphs/                        # Layer 3 output
│   ├── predictions/                   # Layer 4 output
│   ├── explanations/                  # Layer 5 output
│   ├── mtsamples/
│   ├── mtsamples_processed/
│   ├── mtsamples_extracted/
│   ├── mtsamples_graphs/
│   ├── openI/
│   ├── openI_processed/
│   ├── openI_extracted/
│   ├── openI_graphs/
│   ├── bc5cdr/
│   ├── ncbi_disease/
│   └── corpus_labels.csv
│
├── models/
│   ├── lora_weights/
│   │   └── clinicalbert_rel/
│   ├── graph_model/
│   └── readmission_head/
│
├── results/
│   └── ablation_table.csv
│
├── scripts/
│   ├── prepare_mtsamples.py
│   ├── prepare_openI.py
│   └── generate_labels.py
│
├── src/
│   ├── layer1/   (__init__.py, config.py, text_cleaner.py, pipeline.py)
│   ├── layer2/   (__init__.py, config.py, ner_extractor.py, entity_linker.py, relation_extractor.py, temporal_normalizer.py, pipeline.py)
│   ├── layer3/   (__init__.py, config.py, graph_builder.py, node_encoder.py, edge_typer.py, pipeline.py)
│   ├── layer4/   (__init__.py, config.py, clinical_dataset.py, graph_model.py, readmission_head.py, trainer.py, pipeline.py)
│   ├── layer5/   (__init__.py, config.py, feature_explainer.py, report_builder.py, pipeline.py)
│   └── shared/   (__init__.py, constants.py, schema.py, utils.py)
│
├── tests/
│   ├── sample_data/   (sample_discharge_summary.txt, sample_lab_report.txt, sample_prescription.txt)
│   ├── test_layer1.py
│   ├── test_layer2.py
│   ├── test_layer3.py
│   ├── test_layer4.py
│   └── test_layer5.py
│
└── logs/
    └── charta.log
```

---

## 3. Technology Stack & requirements.txt

### Tech Stack Table

| Component | Library | Version | Purpose |
|---|---|---|---|
| Text fixing | `ftfy` | 6.1.3 | Fix encoding issues in plain text files |
| Clinical NER | `scispacy` | 0.5.4 | Biomedical NER + EntityLinker (replaces MedCAT) |
| Relation data | `bioc` | 2.1 | Required to load `bigbio/bc5cdr` RE config |
| Language models | `transformers` | 4.40.0 | ClinicalBERT |
| Deep learning | `torch` | 2.2.2 | Core tensor operations |
| Graph learning | `torch_geometric` | 2.5.2 | GraphSAGE + HeteroData |
| Explainability | `shap` | 0.45.0 | GNN feature attribution (lightweight) |
| Datasets | `datasets` | 2.21.0 | HuggingFace data loader (trust_remote_code supported since v2.18+) |
| Numerics | `numpy` | 1.26.4 | Array ops |
| Data frames | `pandas` | 2.2.2 | CSV handling |
| ML metrics | `scikit-learn` | 1.4.2 | AUROC, F1 |
| Date parsing | `python-dateutil` | 2.9.0 | ISO date normalisation |
| XML parsing | `lxml` | 5.2.1 | Parse OpenI radiology XML |
| HTTP | `requests` | 2.31.0 | Dataset download fallback |
| Validation | `pydantic` | 2.7.0 | Data model validation |
| Progress | `tqdm` | 4.66.2 | Progress bars |
| Testing | `pytest` | 8.0.0 | Unit + integration tests |

### Complete requirements.txt

Save this file as `requirements.txt` at the project root before running `pip install`:

```
# CHARTA v5.0 — complete dependency list
# Python 3.10 required

# ── Layer 1: Clinical Text Preprocessing ─────────────────
ftfy==6.1.3

# ── Layer 2: Clinical NLP ────────────────────────────────
scispacy==0.5.4
bioc==2.1

# ── Language models & fine-tuning ───────────────────────
transformers==4.40.0

# ── Layer 3: Graph ───────────────────────────────────────
torch==2.2.2
torch_geometric==2.5.2
# NOTE: torch_scatter and torch_sparse are installed separately
# in Step 5 using a Windows-compatible wheel URL

# ── Layer 4: GNN Inference ───────────────────────────────
# (no additional dependencies beyond torch_geometric above)

# ── Layer 5: Explainability ──────────────────────────────
shap==0.45.0

# ── Data & ML utilities ──────────────────────────────────
datasets==2.21.0
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
python-dateutil==2.9.0

# ── Parsing & I/O ────────────────────────────────────────
lxml==5.2.1
requests==2.31.0
pydantic==2.7.0
tqdm==4.66.2

# ── Testing ──────────────────────────────────────────────
pytest==8.0.0
```

---

## 4. Dataset Overview & Access Verification

### Access Status Summary

| Dataset | Login? | Approval? | Size | Verified Working? |
|---|---|---|---|---|
| MTSamples (GitHub mirror) | ❌ No | ❌ No | ~10 MB | ✅ Yes |
| BC5CDR NER (`tner/bc5cdr`) | ❌ No | ❌ No | ~5 MB | ✅ Yes |
| BC5CDR RE (`bigbio/bc5cdr`) | ❌ No | ❌ No | ~8 MB | ✅ Yes (needs `pip install bioc`) |
| NCBI Disease | ❌ No | ❌ No | ~2 MB | ✅ Yes |
| OpenI IU-XRay | ❌ No | ❌ No | ~1.5 GB | ✅ Yes |
| PubMedQA | ❌ No | ❌ No | ~300 MB | ✅ Yes (`trust_remote_code=True` required) |
| MedMCQA | ❌ No | ❌ No | ~700 MB | ✅ Yes |

### Dataset Details

**MTSamples** — Primary clinical corpus (Layers 1, 3, 4)
- 4,999 clinical transcriptions across 40 specialties
- Columns used: `description`, `medical_specialty`, `sample_name`, `transcription`, `keywords`
- Primary mirror: `https://raw.githubusercontent.com/eshza/medicalTranscriptsKaggle/master/mtsamples.csv`
- Backup mirror: `https://raw.githubusercontent.com/salgadev/medical-nlp/master/mtsamples.csv`

**BC5CDR** — NER + RE fine-tuning (Layer 2)
- `tner/bc5cdr` — IOB NER tags only → for NER evaluation
- `bigbio/bc5cdr` config `"bc5cdr_bigbio_kb"` → for CID relation extraction → **requires `pip install bioc` first**
- ⚠️ `"bc5cdr_bigbio_re"` config does NOT exist — use `"bc5cdr_bigbio_kb"` only

**NCBI Disease** — NER benchmark evaluation (Layer 2)
- `ncbi/ncbi_disease` — 793 abstracts, pre-split train/val/test

**OpenI IU-XRay** — Multi-visit temporal structure + OCR test images (Layers 1, 3)
- `ykumards/open-i` on HuggingFace, or NIH direct: `https://openi.nlm.nih.gov/imgs/collections/`
- Reports are XML in NLM format (NOT BioC XML — see Bug #10 fix in implementation)
- Each patient `uid` links frontal + lateral images and a structured report with Comparison section

**PubMedQA** — BioGPT fine-tuning (Layer 5)
- ⚠️ `trust_remote_code=True` is REQUIRED — uses custom loading script
- `load_dataset("qiaojin/PubMedQA", "pqa_labeled", trust_remote_code=True)`
 
**MedMCQA** — Factuality evaluation (Layer 5)
- `load_dataset("openlifescienceai/medmcqa")` — no flags needed

### Risk Label Derivation

Labels are derived from document metadata using `scripts/generate_labels.py`:

| Label | HIGH/AT-RISK condition |
|---|---|
| `readmission` | specialty in [Emergency Room, Cardiovascular, Nephrology, Critical Care] OR keywords: readmit, return to ED |
| `deterioration` | keywords: acute, urgent, severe, critical, worsening, ICU transfer, emergent |
| `medication` | 5+ distinct drug names OR keywords: drug interaction, polypharmacy, adverse reaction |

---

## 5. Layer-wise Breakdown

---

### Layer 1 — Clinical Text Preprocessing

**Goal:** Accept plain text (.txt) clinical notes → produce clean, normalised, sentence-segmented text ready for NLP. No NLP here — cleaning and preprocessing only.

#### Pipeline flow

```
TXT Clinical Notes
    ↓
Text Cleaning
    ↓
Normalisation
    ↓
Sentence Segmentation
    ↓
Processed Clinical Text JSON
```

#### Files to create

| File | Role |
|---|---|
| `src/layer1/config.py` | Preprocessing constants and supported extensions |
| `src/layer1/text_cleaner.py` | Normalise, clean, segment |
| `src/layer1/pipeline.py` | Batch-process entire folder of .txt files |
| `run_layer1.py` | CLI entry point |

#### Function-level breakdown

**`config.py`**
```python
SUPPORTED_EXTENSIONS = [".txt"]
MIN_TEXT_LENGTH      = 10      # characters; files below this are treated as empty
```

**`text_cleaner.py`**
```
clean_text(raw_text: str, expand_abbreviations: bool = True) -> dict
  # Returns: {cleaned_text, sentences, sentence_count, original_length, cleaned_length, placeholder_count}
  ├── ftfy.fix_text(text)
  ├── re.sub(r"\[\*\*.*?\*\*\]", "[REDACTED]", text)   # MIMIC-style placeholders
  ├── remove non-printable control characters (keep \n, \t, space)
  ├── collapse multiple spaces/tabs per line
  ├── _should_remove_line(line): True for page numbers, separator lines, single-char lines
  ├── collapse >1 consecutive blank lines to exactly 1
  ├── _expand_abbreviations(text): htn→hypertension, dm→diabetes mellitus, etc.
  └── _segment_sentences(text): split on [.!?] followed by capital letter
```

**`pipeline.py`**
```
run_pipeline(input_folder: str, output_folder: str) -> dict
  # Returns: {processed, failed, empty, skipped, errors, output_files}
  ├── _discover_files(input_path) → list[Path]  (rglob for .txt only)
  ├── for each file_path:
  │     └── _process_single_file(file_path, output_path) → {status, output_path}
  │           ├── guard: 0-byte or < MIN_TEXT_LENGTH chars → status:"empty", skip
  │           ├── read: try utf-8, then latin-1, then cp1252 encoding
  │           ├── clean_text(raw_text)
  │           ├── build output_doc dict (metadata + extraction + cleaning + content)
  │           └── save JSON: {stem}_processed.json
  ├── _save_summary(summary, output_path) → _pipeline_summary.json
  └── _print_summary(summary)
```

#### Expected output schema

```json
{
  "metadata": {
    "source_file": "mtsamples_0001_discharge.txt",
    "file_type": "text",
    "processed_at": "2024-01-15T14:32:07",
    "processing_time_seconds": 0.12,
    "layer": "layer1_clinical_text_preprocessing"
  },
  "extraction": {
    "method": "direct_read",
    "encoding_used": "utf-8",
    "extraction_error": null
  },
  "cleaning": {
    "original_length": 1842,
    "cleaned_length": 1307,
    "sentence_count": 28,
    "placeholder_count": 0
  },
  "content": {
    "cleaned_text": "Patient is a 65-year-old male with hypertension...",
    "sentences": ["Patient is a 65-year-old male.", "He presents with chest pain."],
    "sentence_count": 28
  }
}
```

---

### Layer 2 — Clinical NLP Extraction

**Goal:** Read Layer 1 JSON → extract entities with MeSH/RxNorm concept IDs, clinical relations, and ISO timestamps.

#### Files to create

| File | Role |
|---|---|
| `src/layer2/config.py` | Model names, linker names, thresholds |
| `src/layer2/ner_extractor.py` | ScispaCy NER |
| `src/layer2/entity_linker.py` | MeSH + RxNorm concept linking via scispaCy EntityLinker |
| `src/layer2/relation_extractor.py` | ClinicalBERT CID relation extraction |
| `src/layer2/temporal_normalizer.py` | Date extraction + ISO normalisation |
| `src/layer2/pipeline.py` | Orchestrate all NLP steps |
| `src/layer2/evaluator.py` | NER evaluation (F1) on NCBI Disease test split |
| `run_layer2.py` | CLI entry point |

#### Function-level breakdown

**`config.py`**
```python
from shared.constants import CLINICALBERT_MODEL  # import from shared, not re-defined here

SCISPACY_MODEL     = "en_ner_bc5cdr_md"
# Do NOT use en_core_sci_lg — produces only generic "ENTITY" label, not DISEASE/CHEMICAL
DISEASE_LINKER     = "mesh"
# MeSH: ~30k high-quality disease concepts, zero registration, auto-cached
DRUG_LINKER        = "rxnorm"
# RxNorm: ~100k drug concepts, zero registration, auto-cached
NER_ENTITY_TYPES   = ["DISEASE", "CHEMICAL"]
BC5CDR_NER_DATASET = "tner/bc5cdr"
BC5CDR_RE_DATASET  = "bigbio/bc5cdr"
BC5CDR_RE_CONFIG   = "bc5cdr_bigbio_kb"  # ONLY valid config; "bc5cdr_bigbio_re" does NOT exist
NCBI_DISEASE_DATASET = "ncbi/ncbi_disease"
RELATION_THRESHOLD = 0.6
```

**`ner_extractor.py`** — ⚠️ Fixed: takes `doc + sentence_idx`, NOT raw text
```
load_ner_model() -> spacy.Language
  └── spacy.load(SCISPACY_MODEL)

extract_entities(doc: spacy.Doc, sentence_idx: int) -> list[dict]
  # IMPORTANT: takes a PROCESSED spaCy Doc object, not raw text
  # sentence_idx is the 0-based position of this sentence in the document
  └── for each ent in doc.ents:
        └── yield {
              "text":         ent.text,
              "label":        ent.label_,    # "DISEASE" or "CHEMICAL"
              "start_char":   ent.start_char,
              "end_char":     ent.end_char,
              "sentence_idx": sentence_idx
            }
```

**`entity_linker.py`** — ⚠️ Fixed: uses doc.char_span() not doc[char:char]; ⚠️ Fixed (BUG-N1): linker KB accessed via nlp.get_pipe(), not bare `linker` variable
```
# MedCAT removed (archived Jul 28 2025, all models require NIH login)
# Replacement: scispaCy built-in EntityLinker — zero registration, auto-downloads

add_entity_linkers(nlp: spacy.Language) -> spacy.Language
  # Call ONCE when building the pipeline, not per-sentence
  └── nlp.add_pipe("scispacy_linker", name="mesh_linker",
                   config={"linker_name": DISEASE_LINKER, "resolve_abbreviations": True})
  └── nlp.add_pipe("scispacy_linker", name="rxnorm_linker",
                   config={"linker_name": DRUG_LINKER, "resolve_abbreviations": True},
                   last=True)
  └── return nlp

link_entities(nlp: spacy.Language, doc: spacy.Doc, entities: list[dict]) -> list[dict]
  # ⚠️ nlp must be passed in so we can call nlp.get_pipe(linker_name) to access the KB.
  # Runs AFTER nlp(sentence) — linker already applied inside the spaCy pipeline
  └── for each entity in entities:
        # ⚠️ CORRECT: use doc.char_span() for character offsets, NOT doc[start:end]
        # doc[start:end] uses TOKEN indices — passing char offsets gives wrong spans
        └── span = doc.char_span(
                      entity["start_char"],
                      entity["end_char"],
                      alignment_mode="expand"   # handles tokenisation edge cases
                   )
        └── if span is None:
              log warning; set concept_id=None, concept_name=entity["text"], link_score=0.0
              continue
        └── linker_name = "mesh_linker" if entity["label"]=="DISEASE" else "rxnorm_linker"
        └── kb_ents = span._.kb_ents  # list of (concept_id, score) tuples
        └── if kb_ents:
              concept_id, score = kb_ents[0]   # take top-1 candidate
              # ⚠️ BUG-N1 FIX: retrieve KB via nlp.get_pipe(), not bare `linker` variable
              linker_pipe = nlp.get_pipe(linker_name)
              concept_name = linker_pipe.kb.cui_to_entity[concept_id].canonical_name
            else:
              concept_id, concept_name, score = None, entity["text"], 0.0
        └── entity.update({
              "concept_id":   concept_id,
              "concept_name": concept_name,
              "kb_source":    "mesh" if entity["label"]=="DISEASE" else "rxnorm",
              "link_score":   float(score)
            })
  └── return entities
```

**`relation_extractor.py`**
```
load_relation_model() -> (tokenizer, model)
  # Returns a placeholder until LoRA weights are available.
  └── AutoTokenizer.from_pretrained(CLINICALBERT_MODEL)
  └── AutoModelForSequenceClassification.from_pretrained(
          "models/lora_weights/clinicalbert_rel/"
      )

extract_relations(sentences: list[str], entities: list[dict]) -> list[dict]
  # Returns [] as placeholder if no LoRA weights are present; Layer 3 still builds graphs
  └── for each sentence with 2+ entities in the same sentence_idx:
        └── for each entity pair (e1, e2):
              input = f"[E1] {e1['text']} [/E1] {sentence} [E2] {e2['text']} [/E2]"
              logits = model(tokenize(input))
              label = argmax(logits)  # 0=None, 1=CID
              prob  = softmax(logits)[label]
              if prob >= RELATION_THRESHOLD and label == 1:
                yield {"entity_1": e1["text"], "entity_2": e2["text"],
                       "relation_type": "CID", "confidence": prob,
                       "sentence_idx": e1["sentence_idx"]}
```

**`temporal_normalizer.py`**
```python
TEMPORAL_PATTERNS = [
    r"\b\d{4}-\d{2}-\d{2}\b",                        # ISO: 2023-09-12
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b",
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",                  # 09/12/2023
    r"\b\d{1,2} (?:days?|weeks?|months?) ago\b",
]

extract_temporal_expressions(text: str) -> list[dict]
  └── for each pattern: finditer → {"raw_text", "start_char", "end_char"}

normalize_to_iso(raw_date: str) -> str
  └── dateutil.parser.parse(raw_date, default=datetime(2000,1,1))
  └── return strftime("%Y-%m-%d")
  └── on ValueError/OverflowError: return raw_date + log warning

attach_timestamps_to_entities(entities: list[dict], temporal_exprs: list[dict]) -> list[dict]
  └── for each entity: find nearest temporal_expr in same sentence_idx
  └── entity["timestamp"] = normalized ISO string (or None if none found)
```

**`pipeline.py`**
```
run_pipeline(input_folder: str, output_folder: str) -> dict
  ├── nlp = load_ner_model()
  ├── nlp = add_entity_linkers(nlp)   # loads MeSH + RxNorm KBs once (~1 GB cached)
  ├── tokenizer, rel_model = load_relation_model()
  ├── for each *_processed.json in input_folder:
  │     ├── sentences = doc["content"]["sentences"]
  │     ├── all_entities = []
  │     ├── for idx, sentence in enumerate(sentences):
  │     │     ├── spacy_doc = nlp(sentence)
  │     │     ├── entities  = extract_entities(spacy_doc, idx)   # pass doc + idx
  │     │     ├── entities  = link_entities(nlp, spacy_doc, entities)  # ⚠️ pass nlp for KB access
  │     │     └── all_entities.extend(entities)
  │     ├── relations    = extract_relations(sentences, all_entities)
  │     ├── temp_exprs   = extract_temporal_expressions(" ".join(sentences))
  │     ├── all_entities = attach_timestamps_to_entities(all_entities, temp_exprs)
  │     └── save_json({metadata, entities, relations, temporal_expressions}, output)
  └── return {processed, failed, errors}
```

#### Expected output schema

```json
{
  "metadata": {
    "source_file": "mtsamples_0001_discharge_processed.json",
    "layer": "layer2_nlp_extraction",
    "processed_at": "2024-01-15T14:35:00"
  },
  "entities": [
    {
      "text": "hypertension",
      "label": "DISEASE",
      "start_char": 42,
      "end_char": 54,
      "sentence_idx": 2,
      "concept_id": "D006973",
      "concept_name": "Hypertension",
      "kb_source": "mesh",
      "link_score": 0.91,
      "timestamp": "2023-09-12"
    }
  ],
  "relations": [
    {
      "entity_1": "lisinopril",
      "entity_2": "hypertension",
      "relation_type": "CID",
      "confidence": 0.87,
      "sentence_idx": 5
    }
  ],
  "temporal_expressions": [
    { "raw_text": "September 12, 2023", "normalized": "2023-09-12", "start_char": 18 }
  ]
}
```

---

### Layer 3 — Temporal Document Graph

**Goal:** Transform Layer 2 JSON into a patient-level heterogeneous knowledge graph.

#### Files to create

| File | Role |
|---|---|
| `src/layer3/config.py` | Graph constants |
| `src/layer3/node_encoder.py` | ClinicalBERT embeddings for entity nodes |
| `src/layer3/edge_typer.py` | Build temporal, relation, co-occurrence edges |
| `src/layer3/graph_builder.py` | Assemble HeteroData graph |
| `src/layer3/pipeline.py` | Group by patient, build all graphs |
| `run_layer3.py` | CLI entry point |

#### Function-level breakdown

**`config.py`**
```python
from shared.constants import CLINICALBERT_MODEL  # do NOT re-define here

EMBEDDING_DIM          = 768
CO_OCCUR_WINDOW        = 3     # sentences within 3 of each other = co-occurrence edge
MIN_ENTITIES_PER_GRAPH = 2     # skip patient if fewer than 2 entities found
GRAPH_SAVE_FORMAT      = "pt"
```

**`node_encoder.py`**
```
load_encoder() -> (tokenizer, model)
  └── AutoTokenizer.from_pretrained(CLINICALBERT_MODEL)
  └── AutoModel.from_pretrained(CLINICALBERT_MODEL)
  └── model.eval(); model.to(device)

encode_text(text: str, tokenizer, model) -> np.ndarray
  # Returns shape (768,) — the [CLS] token embedding
  └── inputs = tokenizer(text, max_length=64, truncation=True, return_tensors="pt")
  └── with torch.no_grad(): outputs = model(**inputs)
  └── return outputs.last_hidden_state[0, 0, :].cpu().numpy()

encode_entity_nodes(entities: list[dict], tokenizer, model) -> dict[str, np.ndarray]
  # Key: concept_id if not None, else entity["text"]; Value: 768-dim numpy array
  # ⚠️ BUG-N3 FIX: always use the same key formula as entity_index in graph_builder.py
  #   key = entity["concept_id"] if entity["concept_id"] is not None else entity["text"]
  └── for each unique key:
        text = entity["concept_name"] or entity["text"]
        return {key: encode_text(text, tokenizer, model)}
```

**`edge_typer.py`**
```
build_temporal_edges(visits: list[dict]) -> list[tuple[int,int,str]]
  └── sort visits by their earliest entity timestamp
  └── for (i, i+1) in consecutive pairs: yield (i, i+1, "before")

build_relation_edges(relations: list[dict], entity_index: dict) -> list[tuple]
  └── for each relation: lookup idx of entity_1, entity_2 in entity_index
  └── yield (idx1, idx2, relation["relation_type"])

build_cooccurrence_edges(entities: list[dict], window: int) -> list[tuple]
  └── for each pair (a, b) where abs(a["sentence_idx"] - b["sentence_idx"]) <= window:
        if (a_idx, b_idx) not already in relation edges:
          yield (a_idx, b_idx, "co_occurs_with")
```

**`graph_builder.py`**
```
build_patient_graph(
    extracted_docs: list[dict],
    patient_id: str,
    tokenizer,       # loaded ONCE in pipeline.py — do NOT call load_encoder() here
    encoder_model    # same — calling it per-patient reloads 1.3 GB repeatedly
) -> HeteroData
  ├── collect all entities from all docs into flat list
  ├── build entity_index: {concept_id_or_text → integer_node_idx}
  │   # ⚠️ BUG-N3 FIX: key is (concept_id if concept_id is not None else entity["text"])
  │   # Never use concept_id directly — it may be None for unlinked entities
  ├── build name_index: {concept_id_or_text → human_readable_name}
  │   # Stores entity["concept_name"] (e.g. "Hypertension") for use by Layer 5
  │   # This ensures Layer 5 can display real names, not concept IDs
  ├── build visit_index: {doc_filename → integer_visit_idx}
  ├── node_embeddings = encode_entity_nodes(all_entities, tokenizer, encoder_model)
  ├── # ⚠️ BUG-N3 FIX: use same key logic as entity_index
  ├── entity_x = torch.stack([
  │       tensor(node_embeddings[e["concept_id"] if e["concept_id"] else e["text"]])
  │       for e in entities
  │   ])
  ├── visit_x  = mean-pool entity_x per visit group → torch.Tensor [N_visits, 768]
  ├── patient_x = entity_x.mean(dim=0).unsqueeze(0)  → [1, 768]
  ├── edge_index_occurs_in = entity→visit membership edges
  ├── edge_index_before    = build_temporal_edges(visits)
  ├── edge_index_relates   = build_relation_edges(relations, entity_index)
  ├── edge_index_cooccurs  = build_cooccurrence_edges(entities, CO_OCCUR_WINDOW)
  ├── graph = HeteroData()
  │   graph["entity"].x = entity_x
  │   graph["visit"].x  = visit_x
  │   graph["patient"].x = patient_x
  │   graph["entity","occurs_in","visit"].edge_index = edge_index_occurs_in
  │   graph["visit","before","visit"].edge_index     = edge_index_before
  │   graph["entity","relates_to","entity"].edge_index = edge_index_relates
  │   graph["entity","co_occurs_with","entity"].edge_index = edge_index_cooccurs
  └── return graph, name_index   # name_index saved to graph_meta.json by pipeline.py

validate_graph(graph: HeteroData) -> bool
  ├── assert graph["entity"].x.shape[0] >= MIN_ENTITIES_PER_GRAPH
  ├── assert not torch.isnan(graph["entity"].x).any()
  ├── assert at least 1 edge type has non-empty edge_index
  └── return True / raise ValueError with details
```

**`pipeline.py`**
```
run_pipeline(input_folder: str, output_folder: str) -> dict
  ├── tokenizer, encoder_model = load_encoder()  # load ONCE
  ├── group all *_extracted.json files by patient_id
  │   (patient_id = filename prefix, e.g. "mtsamples_0001" from "mtsamples_0001_*.json")
  ├── for each patient_id, doc_list:
  │     ├── build_patient_graph(doc_list, patient_id, tokenizer, encoder_model)
  │     ├── validate_graph(graph)
  │     ├── torch.save(graph, f"{output_folder}/{patient_id}_graph.pt")
  │     └── save_json(meta, f"{output_folder}/{patient_id}_graph_meta.json")
  └── return {patients_processed, patients_failed, output_files}
```

#### Expected output schema

```
HeteroData(
  entity  = { x: [N_entities, 768] },
  visit   = { x: [N_visits, 768] },
  patient = { x: [1, 768] },
  (entity, occurs_in, visit)       = { edge_index: [2, E1] },
  (visit, before, visit)           = { edge_index: [2, E2] },
  (entity, relates_to, entity)     = { edge_index: [2, E3] },
  (entity, co_occurs_with, entity) = { edge_index: [2, E4] }
)
```

```json
// {patient_id}_graph_meta.json
{
  "patient_id": "mtsamples_0001",
  "num_entities": 24,
  "num_visits": 2,
  "num_edges": 63,
  "entity_index": { "D006973": 0, "D003920": 1 },
  "name_index":   { "D006973": "Hypertension", "D003920": "Diabetes Mellitus" },
  "visit_dates": ["2023-09-12", "2023-10-05"],
  "graph_file": "mtsamples_0001_graph.pt",
  "source_dataset": "mtsamples"
}
```

---

### Layer 4 — Temporal Graph-based Readmission Risk Prediction     ck2

**Goal:** Read temporal patient history graphs from Layer 3 → run a Graph Neural Network → predict 30-day readmission risk as a probability and HIGH/LOW label.

#### Pipeline flow

```
Temporal Patient Graph
        ↓
Graph Neural Network (GraphSAGE)
        ↓
Readmission Risk Prediction
```

#### Files to create

| File | Role |
|---|---|
| `src/layer4/config.py` | GNN dims, training hyperparameters |
| `src/layer4/clinical_dataset.py` | PyTorch Geometric Dataset class |
| `src/layer4/graph_model.py` | ClinicalGraphSAGE model |
| `src/layer4/readmission_head.py` | Single readmission prediction head |
| `src/layer4/trainer.py` | Training loop (runs on Colab T4) |
| `src/layer4/pipeline.py` | Batch inference over all graphs |
| `run_layer4.py` | CLI entry point |

#### Function-level breakdown

**`config.py`**
```python
GRAPHSAGE_HIDDEN_DIM  = 256
GRAPHSAGE_NUM_LAYERS  = 2
RISK_THRESHOLD        = 0.5        # readmission: >= 0.5 → HIGH
LEARNING_RATE         = 2e-4
NUM_EPOCHS            = 20
BATCH_SIZE            = 16
POSITIVE_CLASS_WEIGHT = 3.0        # readmission ~25% of corpus
LABELS_CSV_PATH       = "data/corpus_labels.csv"
# NOTE: No LoRA here — GraphSAGE (~500K params) trained fully end-to-end
# LoRA belongs in Layer 2 (ClinicalBERT) only
```

**`clinical_dataset.py`**
```
class ClinicalGraphDataset(InMemoryDataset):
  __init__(self, graphs_folder: str, labels_csv: str):
    ├── load all *_graph.pt files from graphs_folder
    ├── load labels from labels_csv (columns: patient_id, readmission)
    ├── for each graph: attach .y_readmission as tensor
    └── call self.process()

  __len__(self) -> int
  __getitem__(self, idx: int) -> HeteroData

collate_fn(batch: list[HeteroData]) -> Batch
  # HeteroData CANNOT be stacked by default DataLoader — must use PyG's Batch class
  └── return torch_geometric.data.Batch.from_data_list(batch)
```

**`graph_model.py`**
```
class ClinicalGraphSAGE(torch.nn.Module):
  __init__(self, in_dim=768, hidden_dim=256, out_dim=256, num_layers=2):
    └── self.convs = ModuleList([
          SAGEConv(in_dim, hidden_dim),
          SAGEConv(hidden_dim, out_dim)
        ])
    └── self.bns  = ModuleList([BatchNorm(hidden_dim), BatchNorm(out_dim)])
    └── self.drop = Dropout(0.3)

  forward(self, x_dict, edge_index_dict) -> torch.Tensor
    └── apply each SAGEConv + BatchNorm + ReLU + Dropout on entity nodes
    └── global_mean_pool(entity_x, batch) → [batch_size, 256]

get_patient_embedding(graph: HeteroData, model) -> np.ndarray
  └── model.eval(); with torch.no_grad()
  └── return model(graph.x_dict, graph.edge_index_dict).cpu().numpy()  # shape [256]
```

**`readmission_head.py`**
```
class ReadmissionHead(torch.nn.Module):
  __init__(self, input_dim=256):
    # ⚠️ BUG-N4 FIX: NO Sigmoid here — BCEWithLogitsLoss in trainer.py applies sigmoid
    # internally during training. Adding Sigmoid here causes double-sigmoid, which
    # squashes gradients and prevents convergence. Apply torch.sigmoid() only at
    # inference time in pipeline.py AFTER the model forward pass.
    └── Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→1)   # raw logit output

class ReadmissionRiskModel(torch.nn.Module):
  __init__(self, graph_model: ClinicalGraphSAGE):
    └── self.encoder          = graph_model
    └── self.readmission_head = ReadmissionHead(256)

  forward(self, graph: HeteroData) -> torch.Tensor:
    └── graph_emb = self.encoder(graph)              # [batch, 256]
    └── return self.readmission_head(graph_emb)      # [batch, 1] — raw logit, NOT probability
```

**`trainer.py`**
```
# [Colab] — run on Google Colab T4, not local machine
train(config: dict) -> None
  ├── dataset = ClinicalGraphDataset(graphs_folder, LABELS_CSV_PATH)
  ├── train/val/test split = 80/10/10
  ├── DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)
  ├── model = ReadmissionRiskModel(ClinicalGraphSAGE())
  # ⚠️ Do NOT apply LoRA here — GraphSAGE has no "query"/"value" projection layers
  ├── optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
  ├── criterion = BCEWithLogitsLoss(pos_weight=tensor(POSITIVE_CLASS_WEIGHT))
  ├── scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
  ├── for epoch in range(NUM_EPOCHS):
  │     ├── train_epoch() → compute readmission loss, backward, step
  │     ├── evaluate() → AUROC (sklearn.metrics.roc_auc_score)
  │     ├── scheduler.step()
  │     └── save checkpoint if val_AUROC improved
  └── log final test metrics

evaluate(model, dataloader) -> dict
  └── collect predictions + labels for all batches
  └── {"readmission_auroc": roc_auc_score(y_true, y_pred)}
```

**`pipeline.py`**
```
run_pipeline(input_folder: str, output_folder: str) -> dict
  # Batch inference over ALL *_graph.pt files in input_folder
  ├── model = load ReadmissionRiskModel from models/graph_model/ + models/readmission_head/
  ├── for each *_graph.pt in input_folder:
  │     ├── graph = torch.load(graph_path, weights_only=False)
  │     │   # ⚠️ weights_only=False required: HeteroData is not a plain tensor dict
  │     ├── logit = model(graph)                          # [1] raw logit
  │     ├── # ⚠️ BUG-N4 FIX: apply sigmoid HERE (not in the model) to convert logit → probability
  │     ├── pred  = torch.sigmoid(logit)
  │     ├── risk_score = float(pred.squeeze())
  │     ├── risk_level = "HIGH" if risk_score >= RISK_THRESHOLD else "LOW"
  │     └── save_json(predictions, f"{output_folder}/{patient_id}_predictions.json")
  └── return {"processed": N, "failed": M, "errors": [...]}
```

#### Expected output schema

```json
{
  "metadata": {
    "patient_id": "mtsamples_0001",
    "layer": "layer4_readmission_risk_prediction",
    "predicted_at": "2024-01-15T15:00:00"
  },
  "readmission_risk": 0.84,
  "risk_level": "HIGH"
}
```

---

### Layer 5 — Explainable Clinical Report Generation  ck5

**Goal:** Read Layer 4 readmission predictions + Layer 3 patient graphs → identify the top contributing clinical entities → generate a plain-English, template-based clinical summary.

#### Files to create

| File | Role |
|---|---|
| `src/layer5/config.py` | Explainability settings |
| `src/layer5/feature_explainer.py` | Lightweight feature importance from GNN embeddings |
| `src/layer5/report_builder.py` | Assemble final JSON report from top features |
| `src/layer5/pipeline.py` | Orchestrate explainability and report generation |
| `run_layer5.py` | CLI entry point |

#### Function-level breakdown

**`config.py`**
```python
NUM_TOP_FEATURES = 3    # number of top contributing entities to include in report
```

**`feature_explainer.py`**
```
get_top_features(graph: HeteroData, entity_index: dict, name_index: dict, n: int = NUM_TOP_FEATURES) -> list[dict]
  # Extracts the top-n entities by their embedding magnitude as a lightweight
  # proxy for contribution — no SHAP background corpus needed
  # ⚠️ BUG-N2 FIX: name_index must be passed in from graph_meta.json.
  #   entity_index maps {concept_id_or_text → node_idx}.
  #   reverse_index maps {node_idx → concept_id_or_text} — this is the concept ID, NOT the name.
  #   name_index maps {concept_id_or_text → human_readable_name} (e.g. "D006973" → "Hypertension").
  #   Using reverse_index for entity_name would store "D006973" in entity_name, which is wrong.
  ├── entity_embeddings = graph["entity"].x   # shape [N_entities, 768]
  ├── importance_scores = entity_embeddings.norm(dim=1)  # L2 norm per entity → [N]
  ├── top_indices = argsort(importance_scores, descending=True)[:n]
  ├── reverse_index = {v: k for k, v in entity_index.items()}  # {node_idx → concept_id_or_text}
  └── return [
        {
          "entity_name":  name_index.get(reverse_index.get(idx, ""), f"entity_{idx}"),
          "concept_id":   reverse_index.get(idx),   # concept ID string (e.g. "D006973")
          "importance":   float(importance_scores[idx])
        }
        for idx in top_indices
      ]
```

**`report_builder.py`**
```
format_risk_level(prob: float) -> str
  └── "LOW" if prob < 0.5 else "HIGH"

build_plain_english_summary(risk_score: float, top_features: list[dict]) -> str
  # Template-based — no LLM required
  ├── level = format_risk_level(risk_score)
  ├── factor_names = ", ".join(f["entity_name"] for f in top_features)
  └── return (
        f"{'High' if level=='HIGH' else 'Low'} readmission risk "
        f"({'%.0f' % (risk_score*100)}%) due to {factor_names}."
      )
  # Example: "High readmission risk (84%) due to hypertension, diabetes, and multiple medications."

build_report(patient_id: str, prediction: dict, top_features: list[dict]) -> dict
  └── return {
        "metadata": {
          "patient_id": patient_id,
          "layer": "layer5_explainable_clinical_report"
        },
        "risk_summary": {
          "readmission_risk": prediction["readmission_risk"],
          "risk_level":       prediction["risk_level"]
        },
        "explanation": {
          "plain_english": build_plain_english_summary(prediction["readmission_risk"], top_features),
          "top_factors":   top_features   # uses concept_id, NOT cui
        },
        "disclaimer": "Research prototype — not a substitute for clinical judgment."
      }

save_report(report: dict, output_path: str) -> None
  └── json.dump(report, open(output_path,"w"), ensure_ascii=False, indent=2)
```

**`pipeline.py`**
```
run_pipeline(predictions_folder: str, graphs_folder: str, output_folder: str) -> dict
  ├── for each *_predictions.json in predictions_folder:
  │     ├── prediction = load_json(prediction_file)
  │     ├── graph      = torch.load(corresponding *_graph.pt, weights_only=False)
  │     ├── meta       = load_json(corresponding *_graph_meta.json)
  │     ├── # ⚠️ BUG-N2 FIX: pass both entity_index AND name_index from meta
  │     ├── top_features = get_top_features(graph, meta["entity_index"], meta["name_index"])
  │     ├── report     = build_report(patient_id, prediction, top_features)
  │     └── save_report(report, f"{output_folder}/{patient_id}_report.json")
  └── return summary dict
```

#### Expected output schema

```json
{
  "metadata": {
    "patient_id": "mtsamples_0001",
    "layer": "layer5_explainable_clinical_report"
  },
  "risk_summary": {
    "readmission_risk": 0.84,
    "risk_level": "HIGH"
  },
  "explanation": {
    "plain_english": "High readmission risk (84%) due to hypertension, diabetes, and multiple medications.",
    "top_factors": [
      { "entity_name": "hypertension",  "concept_id": "D006973", "importance": 4.21 },
      { "entity_name": "diabetes",      "concept_id": "D003920", "importance": 3.87 },
      { "entity_name": "metformin",     "concept_id": "D008687", "importance": 3.54 }
    ]
  },
  "disclaimer": "Research prototype — not a substitute for clinical judgment."
}
```

---

## 6. Execution Plan — Step by Step

> **AI Agent rule:** Execute each numbered step completely. Confirm the ✅ VERIFY passes before moving on. Do not skip steps or reorder them.

---

### PHASE 0 — Prerequisites Check (Before Week 1)

```
Step P1  Confirm Python 3.10 installed:
           python --version
           ✅ VERIFY: output starts with "Python 3.10."

Step P2  Confirm Git installed:
           git --version
           ✅ VERIFY: output starts with "git version"

Step P3  Confirm internet access:
           python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=5); print('OK')"
           ✅ VERIFY: prints "OK"

Step P4  Confirm Google Colab accessible:
           Open browser → https://colab.research.google.com → sign in with Google account
           ✅ VERIFY: can create a new notebook and run print("hello")
```

---

### PHASE 1 — Project Setup (Week 1, Day 1)

```
Step 1   Create root folder and navigate into it:
           mkdir CHARTA
           cd CHARTA

Step 2   Create ALL required folders in one block:
           mkdir data\raw\txt
           mkdir data\processed data\extracted data\graphs data\predictions data\explanations
           mkdir data\mtsamples data\mtsamples_processed data\mtsamples_extracted data\mtsamples_graphs
           mkdir data\openI data\openI_processed data\openI_extracted data\openI_graphs
           mkdir data\bc5cdr data\ncbi_disease
           mkdir models\lora_weights\clinicalbert_rel
           mkdir models\graph_model models\readmission_head
           mkdir src\layer1 src\layer2 src\layer3 src\layer4 src\layer5 src\shared
           mkdir scripts tests\sample_data results logs
           ✅ VERIFY: dir src — should list layer1 layer2 layer3 layer4 layer5 shared

Step 3   Create all __init__.py files (makes folders into Python packages):
           echo. > src\__init__.py
           echo. > src\layer1\__init__.py
           echo. > src\layer2\__init__.py
           echo. > src\layer3\__init__.py
           echo. > src\layer4\__init__.py
           echo. > src\layer5\__init__.py
           echo. > src\shared\__init__.py
           echo. > tests\__init__.py
           ✅ VERIFY: dir src\layer1 — should show __init__.py

Step 4   Create .gitignore at project root with this content:
           venv/
           data/mtsamples/
           data/mtsamples_processed/
           data/mtsamples_extracted/
           data/mtsamples_graphs/
           data/openI/
           data/openI_processed/
           data/openI_extracted/
           data/openI_graphs/
           data/bc5cdr/
           data/ncbi_disease/
           data/corpus_labels.csv
           models/graph_model/
           models/readmission_head/
           models/lora_weights/clinicalbert_rel/*.safetensors
           models/lora_weights/clinicalbert_rel/*.bin
           logs/
           *.pt
           *.bin
           *.safetensors
           __pycache__/
           .pytest_cache/

Step 5   Create virtual environment and activate it:
           python -m venv venv
           venv\Scripts\activate
           ✅ VERIFY: command prompt shows (venv) prefix

Step 6   Create requirements.txt with the content from Section 3, then install:
           pip install -r requirements.txt
           ✅ VERIFY: pip show transformers torch scispacy — all show installed versions

Step 7   Install ScispaCy BC5CDR NER model + PyTorch Geometric wheel (Windows):
           pip install scispacy==0.5.4
           pip install "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
           pip install torch_geometric==2.5.2
           pip install torch_scatter torch_sparse -f "https://data.pyg.org/whl/torch-2.2.0+cpu.html"
           ✅ VERIFY: python -c "import spacy; nlp = spacy.load('en_ner_bc5cdr_md'); print('NER model OK')"

Step 8   Warm up scispaCy EntityLinker KBs (downloads ~1 GB, then cached forever):
           python -c "
           import spacy
           from scispacy.linking import EntityLinker
           nlp = spacy.load('en_ner_bc5cdr_md')
           nlp.add_pipe('scispacy_linker', config={'linker_name': 'mesh'})
           print('MeSH KB ready')
           "
           python -c "
           import spacy
           from scispacy.linking import EntityLinker
           nlp = spacy.load('en_ner_bc5cdr_md')
           nlp.add_pipe('scispacy_linker', config={'linker_name': 'rxnorm'})
           print('RxNorm KB ready')
           "
           ✅ VERIFY: both lines print "KB ready" with no errors

Step 9   Verify all critical imports work together:
           python -c "
           import ftfy, spacy, scispacy
           import torch, datasets, torch_geometric, shap, pydantic
           print('torch:', torch.__version__)
           print('datasets:', datasets.__version__)
           print('All imports OK')
           "
           ✅ VERIFY: "All imports OK" printed — datasets version must be 4.x
```

---

### PHASE 2 — Shared Utilities (Week 1, Day 2)

```
Step 10  Create src/shared/constants.py with this EXACT content:
         ────────────────────────────────────────────────────
         CLINICALBERT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
         DISEASE_LINKER     = "mesh"
         DRUG_LINKER        = "rxnorm"
         ────────────────────────────────────────────────────

Step 11  Create src/shared/schema.py with this EXACT content:
         ────────────────────────────────────────────────────
         from pydantic import BaseModel
         from typing import Optional

         class ProcessedDocument(BaseModel):
             source_file:    str
             file_type:      str
             cleaned_text:   str
             sentences:      list[str]
             sentence_count: int

         class ExtractedEntity(BaseModel):
             text:         str
             label:        str
             start_char:   int
             end_char:     int
             sentence_idx: int
             concept_id:   Optional[str]   = None
             concept_name: Optional[str]   = None
             kb_source:    Optional[str]   = None
             link_score:   Optional[float] = None
             timestamp:    Optional[str]   = None
         ────────────────────────────────────────────────────

Step 12  Create src/shared/utils.py with this EXACT content:
         ────────────────────────────────────────────────────
         import json, logging
         from pathlib import Path

         def load_json(path: str) -> dict:
             """Load JSON file and return as dict."""
             with open(path, "r", encoding="utf-8") as f:
                 return json.load(f)

         def save_json(data: dict, path: str) -> None:
             """Save dict to JSON with UTF-8 and pretty indentation."""
             Path(path).parent.mkdir(parents=True, exist_ok=True)
             with open(path, "w", encoding="utf-8") as f:
                 json.dump(data, f, indent=2, ensure_ascii=False)

         def get_logger(name: str) -> logging.Logger:
             """Return logger writing to console (INFO) and logs/charta.log (DEBUG)."""
             logger = logging.getLogger(name)
             if not logger.handlers:
                 logger.setLevel(logging.DEBUG)
                 fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
                 ch = logging.StreamHandler(); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
                 # ⚠️ BUG-N6 FIX: create logs/ directory before opening FileHandler.
                 # If logs/ does not exist when a module is first imported (before Step 2
                 # creates folders), FileHandler raises FileNotFoundError and silently
                 # breaks logging for all subsequent calls in that Python process.
                 Path("logs").mkdir(exist_ok=True)
                 fh = logging.FileHandler("logs/charta.log", encoding="utf-8")
                 fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
                 logger.addHandler(ch); logger.addHandler(fh)
             return logger
         ────────────────────────────────────────────────────

Step 13  Create conftest.py at project root with this EXACT content:
         ────────────────────────────────────────────────────
         import sys
         from pathlib import Path
         sys.path.insert(0, str(Path(__file__).parent / "src"))
         ────────────────────────────────────────────────────
         ✅ VERIFY: python -c "from shared.utils import get_logger; print('shared OK')"
```

---

### PHASE 3 — Layer 1 (Week 1, Day 3)

```
Step 14  Create src/layer1/config.py  (see Layer 1 function spec above)
Step 15  Create src/layer1/text_cleaner.py
Step 16  Create src/layer1/pipeline.py

Step 17  Create run_layer1.py with this EXACT content:
         ────────────────────────────────────────────────────
         import argparse, sys
         from pathlib import Path
         sys.path.insert(0, str(Path(__file__).parent / "src"))
         from layer1.pipeline import run_pipeline

         if __name__ == "__main__":
             parser = argparse.ArgumentParser(description="CHARTA Layer 1 — Clinical Text Preprocessing")
             parser.add_argument("--input",  default="data/raw/txt",   help="Input folder of .txt files")
             parser.add_argument("--output", default="data/processed", help="Output folder")
             args = parser.parse_args()
             summary = run_pipeline(args.input, args.output)
             exit(0 if summary["failed"] == 0 else 1)
         ────────────────────────────────────────────────────

Step 18  Copy 3 sample files into tests/sample_data/:
           Download any 3 MTSamples transcriptions as .txt files from mtsamples.com
           OR create synthetic content matching the format in Section 5, Layer 1 output schema
           ✅ VERIFY: dir tests\sample_data — shows 3 .txt files

Step 19  Run tests: pytest tests/test_layer1.py -v
           ✅ VERIFY: All tests PASSED — zero failures before continuing

Step 20  Run pipeline on sample data:
           python run_layer1.py --input tests/sample_data --output data/processed
           ✅ VERIFY: python -c "
           import json,pathlib
           files = list(pathlib.Path('data/processed').glob('*_processed.json'))
           print(f'{len(files)} files processed')
           doc = json.loads(files[0].read_text())
           assert doc['content']['sentence_count'] > 0
           print('Layer 1 output valid ✅')
           "
```

---

### PHASE 4 — Layer 2 (Weeks 2–3)

```
Step 21  Create tests/test_layer2.py  (see Section 8 for required test cases)
           ✅ VERIFY: python -c "import tests.test_layer2; print('test_layer2 importable OK')"

Step 22  Create tests/sample_data/ clinical text files for Layer 2 testing:
           Copy (or create synthetic) 3 .txt files representing different clinical note types:
             tests/sample_data/sample_discharge_summary.txt
             tests/sample_data/sample_lab_report.txt
             tests/sample_data/sample_prescription.txt
           Each file must contain at least 2 sentences and at least 1 named disease or drug.
           ✅ VERIFY: dir tests\sample_data — shows 3 .txt files

Step 23  Create src/layer2/config.py  (see Layer 2 function spec above)
Step 24  Create src/layer2/ner_extractor.py
           ⚠️  extract_entities(doc: spacy.Doc, sentence_idx: int) — NOT (sentences, nlp)
Step 25  Create src/layer2/entity_linker.py
           ⚠️  use doc.char_span(start, end, alignment_mode="expand") — NOT doc[start:end]
Step 26  Create src/layer2/relation_extractor.py  (returns [] placeholder if no LoRA weights present)
Step 27  Create src/layer2/temporal_normalizer.py
Step 28  Create src/layer2/pipeline.py
           ⚠️  iterate sentences with enumerate(), call extract_entities(doc, idx)

Step 29  Create run_layer2.py with this template:
         ────────────────────────────────────────────────────
         import argparse, sys
         from pathlib import Path
         sys.path.insert(0, str(Path(__file__).parent / "src"))
         from layer2.pipeline import run_pipeline

         if __name__ == "__main__":
             parser = argparse.ArgumentParser(description="CHARTA Layer 2 — NLP Extraction")
             parser.add_argument("--input",   default="data/processed")
             parser.add_argument("--output",  default="data/extracted")
             parser.add_argument("--mode",    default="run",
                                 choices=["run", "eval"])
             parser.add_argument("--dataset", default=None)
             args = parser.parse_args()

             if args.mode == "run":
                 run_pipeline(args.input, args.output)
             elif args.mode == "eval":
                 from layer2.evaluator import evaluate_ner
                 evaluate_ner(args.dataset or "ncbi/ncbi_disease")
         ────────────────────────────────────────────────────

Step 30  Download BC5CDR datasets:
           pip install bioc
           python -c "from datasets import load_dataset; load_dataset('tner/bc5cdr')"
           python -c "from datasets import load_dataset; load_dataset('bigbio/bc5cdr', 'bc5cdr_bigbio_kb')"
           ✅ VERIFY: both print "Downloading..." then complete without errors

Step 31  Download NCBI Disease:
           python -c "from datasets import load_dataset; load_dataset('ncbi/ncbi_disease')"
           ✅ VERIFY: completes without errors

Step 32  Run tests: pytest tests/test_layer2.py -v
           ✅ VERIFY: All tests PASSED

Step 33  Run Layer 2 on Layer 1 output:
           python run_layer2.py --input data/processed --output data/extracted
           ✅ VERIFY: python -c "
           import json, pathlib
           files = list(pathlib.Path('data/extracted').glob('*_extracted.json'))
           print(f'{len(files)} files extracted')
           doc = json.loads(files[0].read_text())
           print('entities found:', len(doc['entities']))
           assert 'concept_id' in doc['entities'][0]
           assert 'cui' not in doc['entities'][0]
           print('Layer 2 output valid ✅')
           "

Step 33b Create src/layer2/evaluator.py locally with this content:
         ────────────────────────────────────────────────────
         """
         src/layer2/evaluator.py
         Evaluates NER F1 on NCBI Disease test split using the loaded scispaCy model.
         """
         from datasets import load_dataset
         from sklearn.metrics import f1_score
         import spacy
         from shared.utils import get_logger
         from layer2.config import SCISPACY_MODEL, NCBI_DISEASE_DATASET

         logger = get_logger(__name__)

         def _iob_entities(token_labels: list[str]) -> set[tuple]:
             """Convert IOB token list to a set of (start, end, type) entity spans."""
             entities, start, cur = set(), None, None
             for i, lbl in enumerate(token_labels):
                 if lbl.startswith("B-"):
                     if cur: entities.add((start, i - 1, cur))
                     start, cur = i, lbl[2:]
                 elif lbl == "O" and cur:
                     entities.add((start, i - 1, cur)); cur = None
             if cur: entities.add((start, len(token_labels) - 1, cur))
             return entities

         def evaluate_ner(dataset_name: str = NCBI_DISEASE_DATASET) -> dict:
             """Compute token-level F1 of scispaCy NER on NCBI Disease test split."""
             logger.info(f"Loading {dataset_name} test split ...")
             ds = load_dataset(dataset_name, split="test")
             nlp = spacy.load(SCISPACY_MODEL)

             all_true, all_pred = [], []
             for example in ds:
                 tokens = example["tokens"]
                 gold   = example["ner_tags"]          # list of int tag ids
                 text   = " ".join(tokens)
                 doc    = nlp(text)
                 pred_bio = ["O"] * len(tokens)
                 for ent in doc.ents:
                     start_tok = len(text[:ent.start_char].split())
                     end_tok   = len(text[:ent.end_char].split())
                     for j in range(start_tok, min(end_tok, len(tokens))):
                         pred_bio[j] = "B-DISEASE" if j == start_tok else "I-DISEASE"
                 gold_bio = [ds.features["ner_tags"].feature.int2str(g) for g in gold]
                 all_true.extend(gold_bio); all_pred.extend(pred_bio)

             f1 = f1_score(all_true, all_pred, average="micro",
                           labels=[l for l in set(all_true) if l != "O"])
             logger.info(f"NER F1 on NCBI Disease test: {f1:.4f}")
             print(f"NER micro-F1: {f1:.4f}  (target > 0.75)")
             return {"ner_f1": f1}
         ────────────────────────────────────────────────────
         ✅ VERIFY (local): python -c "from layer2.evaluator import evaluate_ner; print('evaluator OK')"

Step 35  Evaluate NER on NCBI Disease:
           python run_layer2.py --mode eval --dataset ncbi/ncbi_disease
           ✅ VERIFY: F1 score printed > 0.70 (target > 0.75)
```

---

### PHASE 5 — Layer 3 (Weeks 3–4)

```
Step 36  Create src/layer3/config.py
Step 37  Create src/layer3/node_encoder.py
Step 38  Create src/layer3/edge_typer.py
Step 39  Create src/layer3/graph_builder.py
           ⚠️  encoder loaded ONCE in pipeline.py, passed as argument — not loaded here
Step 40  Create src/layer3/pipeline.py

Step 41  Create run_layer3.py:
         ────────────────────────────────────────────────────
         import argparse, sys
         from pathlib import Path
         sys.path.insert(0, str(Path(__file__).parent / "src"))
         from layer3.pipeline import run_pipeline

         if __name__ == "__main__":
             parser = argparse.ArgumentParser(description="CHARTA Layer 3 — Temporal Graph")
             parser.add_argument("--input",  default="data/extracted")
             parser.add_argument("--output", default="data/graphs")
             args = parser.parse_args()
             run_pipeline(args.input, args.output)
         ────────────────────────────────────────────────────

Step 42  Run tests: pytest tests/test_layer3.py -v
           ✅ VERIFY: All tests PASSED

Step 43  Run Layer 3:
           python run_layer3.py --input data/extracted --output data/graphs
           ✅ VERIFY: python -c "
           import torch, pathlib
           graphs = list(pathlib.Path('data/graphs').glob('*_graph.pt'))
           print(f'{len(graphs)} graphs built')
           g = torch.load(graphs[0], weights_only=False)
           print('entity nodes:', g['entity'].x.shape)
           print('Layer 3 output valid ✅')
           "
```

---

### PHASE 6 — Dataset Preprocessing for Training (Week 5)

```
Step 44  Download MTSamples CSV:
           python -c "
           import requests, pathlib
           pathlib.Path('data/mtsamples').mkdir(exist_ok=True)
           urls = [
               'https://raw.githubusercontent.com/eshza/medicalTranscriptsKaggle/master/mtsamples.csv',
               'https://raw.githubusercontent.com/salgadev/medical-nlp/master/mtsamples.csv'
           ]
           for url in urls:
               try:
                   r = requests.get(url, timeout=30)
                   if r.status_code == 200:
                       open('data/mtsamples/mtsamples.csv','wb').write(r.content)
                       print(f'Downloaded from {url}')
                       break
               except Exception as e:
                   print(f'Failed {url}: {e}')
           "
           ✅ VERIFY: python -c "import pandas as pd; df=pd.read_csv('data/mtsamples/mtsamples.csv'); print(len(df), 'rows')"
           → should print ~4999 rows

Step 45  Download OpenI reports (use HuggingFace, no login):
           python -c "from datasets import load_dataset; ds=load_dataset('ykumards/open-i'); print('OpenI loaded:', len(ds['train']))"
           ✅ VERIFY: prints "OpenI loaded: 3955" or similar

Step 46  Create and run scripts/prepare_mtsamples.py:
           ── Key logic ────────────────────────────────────────────
           import pandas as pd, pathlib, argparse, re

           def read_mtsamples_csv(csv_path: str) -> pd.DataFrame:
               df = pd.read_csv(csv_path, encoding="utf-8")
               df = df[df["transcription"].notna() & (df["transcription"].str.strip() != "")]
               return df.reset_index(drop=True)

           def make_slug(sample_name: str) -> str:
               # No slugify library needed — use built-in str methods only
               return re.sub(r"[^\w]", "_", sample_name.lower())[:60]

           def write_transcription_files(df: pd.DataFrame, output_dir: str) -> None:
               pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
               for idx, row in df.iterrows():
                   slug = make_slug(str(row["sample_name"]))
                   fname = f"mtsamples_{idx:04d}_{slug}.txt"
                   pathlib.Path(output_dir, fname).write_text(
                       row["transcription"], encoding="utf-8", errors="replace")
           ─────────────────────────────────────────────────────────
           python scripts/prepare_mtsamples.py --input data/mtsamples/mtsamples.csv --output data/raw/txt/mtsamples/
           ✅ VERIFY: dir data\raw\txt\mtsamples | find /c ".txt"  → ~4999

Step 47  Create and run scripts/prepare_openI.py:
           ── Key logic ────────────────────────────────────────────
           # OpenI reports are NLM XML format, NOT BioC XML
           # Use lxml to parse the AbstractText elements with Label attributes
           from lxml import etree
           import pathlib, argparse

           SECTIONS = ["COMPARISON", "INDICATION", "FINDINGS", "IMPRESSION"]

           def parse_openI_xml_report(xml_path: str) -> dict:
               tree = etree.parse(xml_path)
               uid  = tree.find(".//uId").get("id", pathlib.Path(xml_path).stem)
               result = {"uid": uid}
               for section in SECTIONS:
                   elem = tree.find(f".//AbstractText[@Label='{section}']")
                   result[section.lower()] = (elem.text or "").strip() if elem is not None else ""
               return result

           def write_report_as_text(report: dict, output_dir: str, visit_num: int = 1) -> None:
               lines = []
               for section in SECTIONS:
                   text = report.get(section.lower(), "")
                   if text:
                       lines.append(f"{section}:\n{text}")
               # ⚠️ ARCHITECTURE FIX: include uid in filename so Layer 3 groups all
               # reports for the same patient by filename prefix (e.g. "openI_CXR123").
               # visit_num allows multiple reports per uid to form multi-visit graphs.
               # Layer 3's grouping logic uses the prefix before the last "_" separator.
               fname = f"openI_{report['uid']}_visit{visit_num:02d}.txt"
               pathlib.Path(output_dir, fname).write_text(
                   "\n\n".join(lines), encoding="utf-8", errors="replace")

           def run(input_dir, txt_output):
               pathlib.Path(txt_output).mkdir(parents=True, exist_ok=True)
               for xml_path in pathlib.Path(input_dir).glob("*.xml"):
                   report = parse_openI_xml_report(str(xml_path))
                   write_report_as_text(report, txt_output, visit_num=1)
           ─────────────────────────────────────────────────────────
           python scripts/prepare_openI.py --input data/openI/ --txt_output data/raw/txt/openI/
           ✅ VERIFY: dir data\raw\txt\openI | find /c ".txt"  → ~3955

Step 48  Create and run scripts/generate_labels.py:
           ── Key logic ────────────────────────────────────────────
           import pandas as pd, re, argparse

           HIGH_RISK_SPECIALTIES = {
               "Emergency Room Reports", "Cardiovascular / Pulmonary",
               "Nephrology", "Critical Care / Intensive Care"
           }
           READMIT_KEYWORDS   = ["readmit", "re-admit", "return to ed", "follow-up in 24 hours"]
           DETERIORATE_WORDS  = ["acute","urgent","severe","critical","worsening",
                                 "icu transfer","emergent","deteriorating"]
           MEDICATION_WORDS   = ["drug interaction","polypharmacy","allergic to",
                                 "adverse reaction","contraindicated"]

           def derive_readmission(row) -> int:
               if row["medical_specialty"] in HIGH_RISK_SPECIALTIES: return 1
               t = str(row["transcription"]).lower()
               return int(any(k in t for k in READMIT_KEYWORDS))

           def derive_deterioration(transcription: str) -> int:
               t = transcription.lower()
               return int(any(k in t for k in DETERIORATE_WORDS))

           def derive_medication(transcription: str) -> int:
               t = transcription.lower()
               if any(k in t for k in MEDICATION_WORDS): return 1
               drug_tokens = re.findall(r"\b[A-Z][a-z]+(?:mab|nib|pril|statin|mycin|cillin)\b", transcription)
               return int(len(set(drug_tokens)) >= 5)

           def run(mtsamples_csv, output_csv):
               df = pd.read_csv(mtsamples_csv, encoding="utf-8")
               df["patient_id"]    = [f"mtsamples_{i:04d}" for i in range(len(df))]
               df["readmission"]   = df.apply(derive_readmission, axis=1)
               df["deterioration"] = df["transcription"].fillna("").apply(derive_deterioration)
               df["medication"]    = df["transcription"].fillna("").apply(derive_medication)
               df[["patient_id","readmission","deterioration","medication"]].to_csv(output_csv, index=False)
               print(f"Labels saved: {output_csv}")
               print(df[["readmission","deterioration","medication"]].sum())
           ─────────────────────────────────────────────────────────
           python scripts/generate_labels.py --mtsamples data/mtsamples/mtsamples.csv --output data/corpus_labels.csv
           ✅ VERIFY: python -c "import pandas as pd; df=pd.read_csv('data/corpus_labels.csv'); print(df.sum())"
           → readmission should be 300–900; if 0 or >3000 check keyword matching

Step 49  Run Layers 1–3 on MTSamples corpus:
           python run_layer1.py --input data/raw/txt/mtsamples --output data/mtsamples_processed
           python run_layer2.py --input data/mtsamples_processed --output data/mtsamples_extracted
           python run_layer3.py --input data/mtsamples_extracted --output data/mtsamples_graphs
           ✅ VERIFY: dir data\mtsamples_graphs | find /c ".pt"  → ~4000+ graph files

Step 50  Run Layers 1–3 on OpenI corpus:
           python run_layer1.py --input data/raw/txt/openI --output data/openI_processed
           python run_layer2.py --input data/openI_processed --output data/openI_extracted
           python run_layer3.py --input data/openI_extracted --output data/openI_graphs
           ✅ VERIFY: dir data\openI_graphs | find /c ".pt"  → ~3000+ graph files

Step 51  Verify combined corpus size is adequate:
           python -c "
           import pathlib
           mt = len(list(pathlib.Path('data/mtsamples_graphs').glob('*_graph.pt')))
           oi = len(list(pathlib.Path('data/openI_graphs').glob('*_graph.pt')))
           print(f'MTSamples graphs: {mt}  OpenI graphs: {oi}  Total: {mt+oi}')
           assert mt + oi >= 3000, 'Need at least 3000 graphs for training'
           print('Corpus size adequate ✅')
           "
```

---

### PHASE 7 — Layer 4: GNN Training (Weeks 6–8) [Colab]

```
Step 52  Create src/layer4/config.py
Step 53  Create src/layer4/clinical_dataset.py
Step 54  Create src/layer4/graph_model.py
Step 55  Create src/layer4/readmission_head.py
Step 56  Create src/layer4/trainer.py
           ⚠️  Do NOT apply LoRA here — GraphSAGE has no "query"/"value" layers
           ⚠️  Use BCEWithLogitsLoss, NOT BCELoss (includes sigmoid — avoids double sigmoid)
Step 57  Create src/layer4/pipeline.py
           ⚠️  torch.load(path, weights_only=False) — required for HeteroData objects

Step 58  Create run_layer4.py with mode support:
         ────────────────────────────────────────────────────
         import argparse, sys
         from pathlib import Path
         sys.path.insert(0, str(Path(__file__).parent / "src"))

         if __name__ == "__main__":
             parser = argparse.ArgumentParser(description="CHARTA Layer 4 — Readmission Risk Prediction")
             parser.add_argument("--mode", choices=["train","eval","run"], default="run")
             parser.add_argument("--graphs",  nargs="+", default=["data/mtsamples_graphs","data/openI_graphs"])
             parser.add_argument("--input",   default="data/graphs")
             parser.add_argument("--predictions", default="data/predictions")
             args = parser.parse_args()

             if args.mode == "train":
                 from layer4.trainer import train
                 train({})
             elif args.mode == "eval":
                 from layer4.trainer import evaluate_on_test
                 evaluate_on_test()
             elif args.mode == "run":
                 from layer4.pipeline import run_pipeline
                 run_pipeline(args.input, args.predictions)
         ────────────────────────────────────────────────────

Step 59  [Colab] Upload training data and run training:
           # In Colab:
           # Mount Google Drive or upload data/mtsamples_graphs/, data/openI_graphs/,
           # data/corpus_labels.csv, and the src/ folder
           !pip install torch_geometric torch peft accelerate
           !python run_layer4.py --mode train
           # Expected: 2–4 hours on free T4 GPU
           # Download models/graph_model/ and models/readmission_head/ back to local machine
           ✅ VERIFY: dir models\graph_model — shows checkpoint files

Step 60  Run evaluation on held-out test split:
           python run_layer4.py --mode eval
           ✅ VERIFY: readmission AUROC > 0.65
           → If AUROC < 0.55 (barely above random): check that corpus_labels.csv has correct
             label distribution; verify graph embedding shapes are consistent

Step 61  Run tests: pytest tests/test_layer4.py -v
           ✅ VERIFY: All tests PASSED
```

---

### PHASE 8 — Layer 5 (Weeks 7–8)

```
Step 62  Create src/layer5/config.py
Step 63  Create src/layer5/feature_explainer.py
Step 64  Create src/layer5/report_builder.py
Step 65  Create src/layer5/pipeline.py

Step 66  Create run_layer5.py:
         ────────────────────────────────────────────────────
         import argparse, sys
         from pathlib import Path
         sys.path.insert(0, str(Path(__file__).parent / "src"))

         if __name__ == "__main__":
             parser = argparse.ArgumentParser(description="CHARTA Layer 5 — Explainable Clinical Report")
             parser.add_argument("--predictions", default="data/predictions")
             parser.add_argument("--graphs",      default="data/graphs")
             parser.add_argument("--output",      default="data/explanations")
             args = parser.parse_args()

             from layer5.pipeline import run_pipeline
             run_pipeline(args.predictions, args.graphs, args.output)
         ────────────────────────────────────────────────────

Step 67  Run tests: pytest tests/test_layer5.py -v
           ✅ VERIFY: All tests PASSED

Step 68  Run Layer 5:
           python run_layer5.py --predictions data/predictions --graphs data/graphs --output data/explanations
           ✅ VERIFY: python -c "
           import json, pathlib
           reports = list(pathlib.Path('data/explanations').glob('*_report.json'))
           print(f'{len(reports)} reports generated')
           r = json.loads(reports[0].read_text())
           assert r['explanation']['plain_english'] != ''
           assert 'concept_id' in r['explanation']['top_factors'][0]
           assert 'cui' not in r['explanation']['top_factors'][0]
           print('Layer 5 output valid ✅')
           "
```

---

### PHASE 9 — End-to-End Pipeline (Week 9)

```
Step 69  Create run_pipeline.py with this EXACT content:
         ────────────────────────────────────────────────────
         """
         run_pipeline.py — CHARTA full end-to-end pipeline
         Processes a folder of plain text clinical notes through all 5 layers.
         """
         import argparse, sys, time
         from pathlib import Path
         sys.path.insert(0, str(Path(__file__).parent / "src"))

         from layer1.pipeline import run_pipeline as l1
         from layer2.pipeline import run_pipeline as l2
         from layer3.pipeline import run_pipeline as l3
         from layer4.pipeline import run_pipeline as l4
         from layer5.pipeline import run_pipeline as l5

         def run_full_pipeline(input_folder: str) -> None:
             start = time.time()
             print(f"\n{'='*60}")
             print("CHARTA Full Pipeline Starting")
             print(f"Input: {input_folder}")
             print(f"{'='*60}\n")

             print("[Layer 1] Clinical Text Preprocessing...")
             l1(input_folder, "data/processed")

             print("[Layer 2] Clinical NLP Extraction...")
             l2("data/processed", "data/extracted")

             print("[Layer 3] Temporal Patient Graph...")
             l3("data/extracted", "data/graphs")

             print("[Layer 4] Readmission Risk Prediction...")
             l4("data/graphs", "data/predictions")

             print("[Layer 5] Explainable Clinical Report...")
             l5("data/predictions", "data/graphs", "data/explanations")

             elapsed = round(time.time() - start, 1)
             print(f"\n{'='*60}")
             print(f"Pipeline complete in {elapsed}s")
             print(f"Reports in: data/explanations/")
             print(f"{'='*60}\n")

         if __name__ == "__main__":
             parser = argparse.ArgumentParser(description="CHARTA End-to-End Pipeline")
             parser.add_argument("--input", default="data/raw/txt", help="Folder of plain text clinical notes")
             args = parser.parse_args()
             run_full_pipeline(args.input)
         ────────────────────────────────────────────────────

Step 70  Test end-to-end pipeline on 3 sample documents:
           python run_pipeline.py --input tests/sample_data
           ✅ VERIFY: python -c "
           import pathlib, json
           reports = list(pathlib.Path('data/explanations').glob('*_report.json'))
           print(f'{len(reports)} reports generated end-to-end')
           assert len(reports) > 0
           print('End-to-end pipeline OK ✅')
           "
```

---

### PHASE 10 — Ablation + Evaluation (Weeks 10–11)

```
Step 71  Ablation A — No temporal edges:
           Modify Layer 3 edge_typer.py to return [] from build_temporal_edges()
           Re-run Layers 3+4; record AUROC
           ✅ RECORD: save to results/ablation_table.csv row "no_temporal"

Step 72  Ablation B — No entity linking (concept_id = None for all entities):
           Modify Layer 2 entity_linker.py to return concept_id=None for every entity
           Re-run Layers 2+3+4; record AUROC
           ✅ RECORD: save to ablation_table.csv row "no_entity_linking"

Step 73  Verify ablation results saved:
           python -c "import pandas as pd; df=pd.read_csv('results/ablation_table.csv'); print(df)"
```

---

### PHASE 11 — Paper and Submission (Weeks 12–16)

```
Step 74  Write paper sections in order:
           Abstract → Introduction → Related Work → Method → Experiments → Results → Conclusion
           ✅ TARGET: ~8 pages for workshop papers, ~12 for full conference

Step 75  Generate all figures:
           Figure 1: CHARTA system architecture (5-layer pipeline diagram)
           Figure 2: Readmission AUROC curve (held-out test set)
           Figure 3: Ablation study bar chart from results/ablation_table.csv
           Figure 4: Sample patient report with explainable clinical summary

Step 76  Prepare GitHub repository:
           git init
           git add .
           git commit -m "CHARTA v5.0 — MSc AI Project"
           Push to GitHub (check .gitignore excludes all datasets and model weights)
           Add README.md with: abstract, requirements, dataset download steps, usage examples
           ✅ VERIFY: clone the repo in a fresh folder; verify all dataset download steps work

Step 77  Submit:
           Target venue 1: ACL BioNLP Workshop (https://aclanthology.org/venues/bionlp/)
           Target venue 2: EMNLP Clinical NLP Workshop
           Target venue 3: IEEE JBHI (journal, longer review cycle)
           OR: MSc thesis portal (check university submission deadline)
```

---

## 7. Constraints and Rules

### Code Design Rules

| Rule | Requirement |
|---|---|
| **Language** | Python 3.10 exactly |
| **Function length** | Max 40 lines per function — split if longer |
| **Import direction** | Layer N imports only from Layer N or `src/shared/`. Never from Layer M > N |
| **Configuration** | Every path, model name, and threshold lives in that layer's `config.py` — never inline |
| **Logging** | Use `get_logger(__name__)` from shared/utils.py — never use `print()` in production code |
| **Error handling** | All file I/O, model calls, and dataset loads inside `try/except` — errors returned in result dict, never swallowed silently |
| **Type hints** | Every function parameter and return value must have type hints |
| **Docstrings** | Every function: one-line minimum docstring |
| **Return shape** | Every pipeline function returns `{"status":"success"/"failed", "error": None/str, ...}` |
| **Tests** | Every function with logic has ≥ 1 test. No untested production code |
| **No runtime installs** | Never call `pip install` inside a Python function — all deps in `requirements.txt` |

### Architecture Rules

- **Layer isolation:** layers communicate via JSON files for metadata and `.pt` PyTorch files for graph tensors. No direct cross-layer function calls.
  - Layer 1 → Layer 2: `*_processed.json` (JSON)
  - Layer 2 → Layer 3: `*_extracted.json` (JSON)
  - Layer 3 → Layer 4: `*_graph.pt` (PyTorch) + `*_graph_meta.json` (JSON)
  - Layer 4 → Layer 5: `*_predictions.json` (JSON); Layer 5 also reads `*_graph.pt` and `*_graph_meta.json` from Layer 3
- **Shared code:** anything used by 2+ layers goes in `src/shared/`. Never copy-paste between layers.
- **Model loading:** load models ONCE at the start of a pipeline run. Never inside per-document or per-patient loops.
- **No global mutable state:** never use module-level variables that change at runtime. Pass everything through function arguments.
- **`weights_only=False` on torch.load:** required for HeteroData objects — PyTorch 2.0+ raises FutureWarning without it, will error in 2.4+.

---

## 8. Testing Strategy

### Test coverage requirements

| Test file | What it must test |
|---|---|
| `test_layer1.py` | Text cleaning (placeholders, abbreviations, segmentation), empty file handling, non-UTF-8 encoding fallback, pipeline batch summary |
| `test_layer2.py` | NER output format, entity linking returns concept_id not cui, char_span alignment, relation extraction threshold filter, temporal ISO normalisation |
| `test_layer3.py` | Graph node count ≥ MIN_ENTITIES, edge types all present, embedding shape [N,768], no NaN values, validate_graph raises on bad input |
| `test_layer4.py` | Model forward pass output shape [batch,1], risk scores in [0,1] only AFTER sigmoid applied in pipeline (model outputs raw logits), risk_level HIGH/LOW label correct, torch.load with weights_only=False, confirm no Sigmoid in ReadmissionHead layers |
| `test_layer5.py` | top_factors use concept_id not cui, entity_name is human-readable string not a concept ID code, plain_english string non-empty, report JSON has all required keys, importance scores are positive floats, get_top_features requires name_index argument |

### Run commands

```cmd
REM All tests
pytest tests/ -v --tb=short

REM One layer
pytest tests/test_layer2.py -v

REM With coverage
pytest tests/ --cov=src --cov-report=term-missing
```

---

## 9. Evaluation Metrics

| Component | Metric | Target | Dataset | Notes |
|---|---|---|---|---|
| NER (Layer 2) | F1-score on NCBI Disease test | > 0.75 | NCBI Disease | Standard benchmark |
| Entity Linking (L2) | Accuracy@1 (top-1 concept correct) | > 0.70 | MTSamples spot-check (20 manual) | MeSH precision |
| Relation Extraction (L2) | Micro-F1 on BC5CDR test | > 0.65 | BC5CDR | ClinicalBERT with LoRA |
| Readmission Risk (L4) | AUROC on held-out test | > 0.65 | MTSamples + OpenI | Random = 0.50 |
| Explanation Coverage (L5) | % reports with ≥ 3 named entities | > 0.90 | MTSamples predictions | Template quality check |

---

## 10. Known Bug Registry

All bugs identified across v1.0–v2.4 and fixed in v3.0:

| ID | Severity | Location | Bug | Fix |
|---|---|---|---|---|
| B1 | High | Layer 2 `ner_extractor.py` | `extract_entities` signature took `(sentences, nlp)` but pipeline called with `(doc, idx)` | Signature fixed to `(doc: spacy.Doc, sentence_idx: int)` |
| B2 | High | Layer 2 `entity_linker.py` | `doc[char:char]` used token indices to index by character offset — wrong spans | Replaced with `doc.char_span(start, end, alignment_mode="expand")` |
| B3 | High | Layer 5 JSON output | `"cui"` field used for 2 entities, `"concept_id"` for the third — inconsistent | All entities now use `"concept_id"` exclusively |
| B4 | High | Layer 5 `shap_explainer.py` | `get_top_shap_features` returned `"cui"` key | Changed to `"concept_id"` |
| B5 | High | Tech stack table | `datasets==2.19.0` — too old; `trust_remote_code` not supported until v3.x | Updated to `datasets==2.21.0` (trust_remote_code supported since v2.18+; v4.0.0 does not exist on PyPI) |
| B6 | Medium | `scripts/prepare_mtsamples.py` | `slugify()` called without installing `python-slugify` library | Replaced with `re.sub(r"[^\w]","_",name.lower())[:60]` |
| B7 | High | Layer 4 `risk_heads.py` | `self.rag_projection = Linear(256, 256)` — wrong input dim; RAG context is 768-dim | Fixed to `Linear(768, 256)` |
| B8 | High | Layer 4 `pipeline.py` | `torch.load(path)` — deprecated in PyTorch 2.0, error in 2.4+ | Added `weights_only=False` |
| B9 | Medium | Layer 2 step 8 | Referenced "Step 34" for LoRA fine-tuning — step did not exist in plan | Step 34 now explicitly defined in Phase 4 |
| B10 | Medium | `scripts/prepare_openI.py` | Described loading "BioC XML" format — OpenI uses NLM XML, not BioC | Corrected to use `lxml.etree` with NLM-format `AbstractText[@Label]` xpath |
| B11 | High | MedCAT (all versions) | MedCAT repository archived Jul 28 2025; all models require NIH registration | Replaced with scispaCy built-in `EntityLinker` (mesh + rxnorm) |
| B12 | High | BC5CDR config name | `"bc5cdr_bigbio_re"` does not exist in the dataset | Corrected to `"bc5cdr_bigbio_kb"` |
| B13 | High | ScispaCy S3 URL | `ai2-s3-scispacy` bucket name wrong | Corrected to `ai2-s2-scispacy` |
| B14 | High | PubMedQA | `load_dataset("qiaojin/PubMedQA")` fails with datasets v4.0+ | Added `trust_remote_code=True` everywhere |
| B15 | Medium | MTSamples download | Single GitHub mirror (personal repo, 0 stars) — could disappear | Added `salgadev/medical-nlp` as backup mirror with Python fallback |
| BUG-N1 | High | Layer 2 `entity_linker.py` | `link_entities()` called `linker.kb.cui_to_entity[concept_id]` but `linker` was never defined in that scope — NameError at runtime, Layer 2 completely non-functional | Fixed: retrieve linker via `nlp.get_pipe(linker_name)` and pass `nlp` as first argument to `link_entities()` |
| BUG-N2 | High | Layer 5 `feature_explainer.py` | `entity_name` was set to `reverse_index.get(idx)`, which returns the concept_id key (e.g. "D006973"), not a human-readable name — reports showed IDs instead of names like "Hypertension" | Fixed: added `name_index` field to `graph_meta.json` (Layer 3), passed to `get_top_features()` and used for `entity_name` lookup |
| BUG-N3 | High | Layer 3 `graph_builder.py` + `node_encoder.py` | `entity_x` used `e["concept_id"]` as the lookup key, causing `KeyError` whenever any entity had `concept_id=None` (unlinked entities, expected behaviour) — crash on any patient with a failed entity link | Fixed: use `e["concept_id"] if e["concept_id"] else e["text"]` as key in both `encode_entity_nodes()` and `build_patient_graph()` |
| BUG-N4 | Critical | Layer 4 `readmission_head.py` + `trainer.py` | `ReadmissionHead` ended with `Sigmoid`, but `trainer.py` used `BCEWithLogitsLoss` which applies sigmoid internally — double-sigmoid squashes all gradients, model cannot converge | Fixed: removed `Sigmoid` from `ReadmissionHead`; model now outputs raw logits; `torch.sigmoid()` applied only in `pipeline.py` at inference time |
| BUG-N5 | Medium | `logs/` directory in `utils.py` | `get_logger()` opened `FileHandler("logs/charta.log")` before `logs/` directory existed — if any module was imported before the project setup steps created the folder, Python raised `FileNotFoundError` and silently disabled all file logging | Fixed: added `Path("logs").mkdir(exist_ok=True)` inside `get_logger()` before creating the FileHandler |
| BUG-N6 | Medium | `.gitignore` | `models/` was excluded entirely, which also excluded `models/lora_weights/clinicalbert_rel/adapter_config.json` — this config file records the LoRA architecture and is required to load the model; cloning the repo without it makes the model unloadable | Fixed: replaced `models/` exclusion with specific exclusions for `*.safetensors` and `*.bin`; `adapter_config.json` is now tracked in git |

---

*End of CHARTA AI Agent Implementation Guide v6.0*
*Bug fixes in v6.0: BUG-N1 (undefined linker variable in entity_linker.py), BUG-N2 (entity_name showed concept ID instead of human-readable name in Layer 5), BUG-N3 (KeyError on None concept_id in graph_builder.py and node_encoder.py), BUG-N4 (double-sigmoid causing training non-convergence in readmission_head.py + trainer.py), BUG-N5 (FileNotFoundError for logs/ directory in utils.py), BUG-N6 (adapter_config.json excluded from git by overbroad .gitignore). Additionally fixed: datasets version pin corrected from non-existent 4.0.0 to 2.21.0; missing steps 21–22 added; Architecture Rules updated to document .pt inter-layer communication; prepare_openI.py filename format fixed for multi-visit patient grouping in Layer 3; name_index added to graph_meta.json schema.*
*v5.0: Simplified Layer 1 to plain text (.txt) only; Layer 4 predicts readmission risk only via GNN without RAG/FAISS; Layer 5 uses template-based explainable report generation without BioGPT or SHAP complexity.
Execution plan restructured for AI agent step-by-step execution with ✅ VERIFY checkpoints at every step.
All runnable code stubs provided for shared utilities, runner scripts, and data preparation scripts.*
