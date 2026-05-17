 # CHARTA — AI Agent Implementation Guide
### Clinical History-Aware RAG-augmented Temporal Architecture
> **Version:** 4.0 | **Target:** MSc Final Year Project | **Optimised for:** AI-assisted development
> **Changelog v4.0:** Fixed top-level import crash in run_layer2.py (B16); added missing trainer.py/evaluator.py/finetuner.py to file creation lists (B17); corrected PEFT 0.10.0 safetensors verify path (B18); fully expanded all three Google Colab fine-tuning steps (Steps 34, 62, 71) with notebook cells, Drive mount, and download instructions.

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

**Tesseract OCR (external binary — NOT a pip package):**
```cmd
REM Download installer: https://github.com/UB-Mannheim/tesseract/wiki
REM Run UB Mannheim installer → install to: C:\Program Files\Tesseract-OCR\
REM After install, open NEW terminal and verify:
tesseract --version
REM Must print: tesseract 5.x.x
REM If "not recognised": add C:\Program Files\Tesseract-OCR\ to Windows System PATH
```

> **How to add to PATH on Windows:**
> Search "Environment Variables" → System Properties → Advanced → Environment Variables
> → Under "System variables" find "Path" → Edit → New → paste `C:\Program Files\Tesseract-OCR\`
> → OK → OK → OK → close and reopen terminal.

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
- Prior knowledge of GNNs, Transformers, or FAISS
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

Run this before starting Step 1. All four lines must succeed:

```cmd
python --version
REM → Python 3.10.x

tesseract --version
REM → tesseract 5.x.x

git --version
REM → git version 2.x.x

python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=5); print('Internet OK')"
REM → Internet OK
```

If all four pass: proceed to Step 1.

---

## 1. Project Overview

CHARTA is an end-to-end clinical AI system that:

- **Ingests** raw medical documents (PDFs, scanned images, plain text)
- **Extracts** clinical entities (diagnoses, drugs, lab values) using biomedical NLP
- **Builds** a patient-level temporal knowledge graph across multiple visits
- **Predicts** clinical risk (readmission, deterioration, medication risk) using a GNN + RAG pipeline
- **Explains** every risk score in plain English using counterfactual generation

**Input:** Folder of unstructured medical documents
**Output:** Risk score + patient summary + plain-language explanation

### Architecture in one line

```
Raw docs → [L1 Ingest] → [L2 NLP] → [L3 Graph] → [L4 Risk] → [L5 Explain] → Patient report
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
│   │   ├── pdfs/
│   │   ├── images/
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
│   ├── corpus_index/
│   ├── corpus_labels.csv
│   └── synthetic_explanations.json
│
├── models/
│   ├── lora_weights/
│   │   ├── clinicalbert_rel/
│   │   └── biogpt_explainer/
│   ├── graph_model/
│   └── risk_heads/
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
│   ├── layer1/   (__init__.py, config.py, pdf_extractor.py, image_extractor.py, text_cleaner.py, pipeline.py)
│   ├── layer2/   (__init__.py, config.py, ner_extractor.py, entity_linker.py, relation_extractor.py, temporal_normalizer.py, pipeline.py)
│   ├── layer3/   (__init__.py, config.py, graph_builder.py, node_encoder.py, edge_typer.py, pipeline.py)
│   ├── layer4/   (__init__.py, config.py, clinical_dataset.py, faiss_indexer.py, rag_retriever.py, graph_model.py, risk_heads.py, trainer.py, pipeline.py)
│   ├── layer5/   (__init__.py, config.py, shap_explainer.py, counterfactual_generator.py, report_builder.py, pipeline.py)
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
| PDF parsing | `pdfplumber` | 0.10.3 | Native text from PDFs |
| PDF rendering | `PyMuPDF` | 1.24.0 | Render scanned pages → images (no Poppler needed) |
| OCR | `pytesseract` | 0.3.10 | Python wrapper for Tesseract |
| Image processing | `opencv-python` | 4.9.0.80 | Image preprocessing for OCR |
| Text fixing | `ftfy` | 6.1.3 | Fix encoding issues |
| Image formats | `Pillow` | 10.3.0 | PIL Image support for OCR pipeline |
| Clinical NER | `scispacy` | 0.5.4 | Biomedical NER + EntityLinker (replaces MedCAT) |
| Relation data | `bioc` | 2.1 | Required to load `bigbio/bc5cdr` RE config |
| Language models | `transformers` | 4.40.0 | ClinicalBERT / BioGPT |
| PEFT / LoRA | `peft` | 0.10.0 | Parameter-efficient fine-tuning |
| Acceleration | `accelerate` | 0.29.3 | Required by HuggingFace PEFT |
| Deep learning | `torch` | 2.2.2 | Core tensor operations |
| Graph learning | `torch_geometric` | 2.5.2 | GraphSAGE + HeteroData |
| Vector search | `faiss-cpu` | 1.8.0 | Patient similarity retrieval |
| Explainability | `shap` | 0.45.0 | GNN feature attribution |
| Datasets | `datasets` | 4.0.0 | HuggingFace data loader (v4 required for trust_remote_code) |
| Numerics | `numpy` | 1.26.4 | Array ops |
| Data frames | `pandas` | 2.2.2 | CSV handling |
| ML metrics | `scikit-learn` | 1.4.2 | AUROC, F1 |
| Date parsing | `python-dateutil` | 2.9.0 | ISO date normalisation |
| Text metrics | `rouge-score` | 0.1.2 | ROUGE for explanation eval |
| Text metrics | `bert-score` | 0.3.13 | BERTScore for explanation eval |
| XML parsing | `lxml` | 5.2.1 | Parse OpenI radiology XML |
| HTTP | `requests` | 2.31.0 | Dataset download fallback |
| Validation | `pydantic` | 2.7.0 | Data model validation |
| Progress | `tqdm` | 4.66.2 | Progress bars |
| Testing | `pytest` | 8.0.0 | Unit + integration tests |

### Complete requirements.txt

Save this file as `requirements.txt` at the project root before running `pip install`:

```
# CHARTA v3.0 — complete dependency list
# Python 3.10 required

# ── Layer 1: Document Ingestion ──────────────────────────
pdfplumber==0.10.3
PyMuPDF==1.24.0
pytesseract==0.3.10
opencv-python==4.9.0.80
ftfy==6.1.3
Pillow==10.3.0

# ── Layer 2: Clinical NLP ────────────────────────────────
scispacy==0.5.4
bioc==2.1

# ── Language models & fine-tuning ───────────────────────
transformers==4.40.0
peft==0.10.0
accelerate==0.29.3

# ── Layer 3: Graph ───────────────────────────────────────
torch==2.2.2
torch_geometric==2.5.2
# NOTE: torch_scatter and torch_sparse are installed separately
# in Step 5 using a Windows-compatible wheel URL

# ── Layer 4: RAG + Inference ─────────────────────────────
faiss-cpu==1.8.0

# ── Layer 5: Explainability ──────────────────────────────
shap==0.45.0

# ── Data & ML utilities ──────────────────────────────────
datasets==4.0.0
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
python-dateutil==2.9.0

# ── Evaluation ───────────────────────────────────────────
rouge-score==0.1.2
bert-score==0.3.13

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

### Layer 1 — Document Ingestion

**Goal:** Accept any medical document format → clean UTF-8 text ready for NLP. No NLP here — extraction and cleaning only.

#### Files to create

| File | Role |
|---|---|
| `src/layer1/config.py` | Tesseract path, OCR constants |
| `src/layer1/pdf_extractor.py` | PDF text extraction (native + OCR fallback via PyMuPDF) |
| `src/layer1/image_extractor.py` | OCR from JPG/PNG medical scans |
| `src/layer1/text_cleaner.py` | Normalise, clean, segment |
| `src/layer1/pipeline.py` | Batch-process entire folder |
| `run_layer1.py` | CLI entry point |

#### Function-level breakdown

**`config.py`**
```python
import sys

# OS-aware Tesseract path — do NOT hardcode for Linux/Mac
if sys.platform == "win32":
    TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    TESSERACT_CMD = "tesseract"  # on PATH for Linux/Mac

MIN_NATIVE_CHARS = 20       # chars below this → treat page as scanned
PDF_RENDER_DPI   = 300      # DPI for PyMuPDF page rendering
TESSERACT_CONFIG = "--psm 3 --oem 3"
SUPPORTED_EXTENSIONS = {
    "pdf":   [".pdf"],
    "image": [".jpg", ".jpeg", ".png", ".tiff", ".bmp"],
    "text":  [".txt"],
}
```

**`pdf_extractor.py`**
```
extract_text_from_pdf(pdf_path: str) -> dict
  # Returns: {file_name, total_pages, pages: list[dict], full_text, error}
  ├── guard: file must exist and size > 0 bytes
  ├── with pdfplumber.open(pdf_path) as pdf:
  │     └── for each page: _extract_single_page(page, page_num, pdf_path)
  └── join pages with "\n\n" → full_text

_extract_single_page(page, page_num: int, pdf_path: str) -> dict
  # Returns: {page_num, method, text, char_count}
  ├── native_text = page.extract_text() or ""
  ├── if len(native_text) >= MIN_NATIVE_CHARS: return {method:"native", text}
  └── else: return _ocr_pdf_page_with_fitz(pdf_path, page_num)

_ocr_pdf_page_with_fitz(pdf_path: str, page_num: int) -> dict
  # Uses PyMuPDF (fitz) — NO Poppler dependency
  ├── doc = fitz.open(pdf_path)
  ├── page = doc[page_num - 1]   # fitz is 0-indexed
  ├── mat = fitz.Matrix(PDF_RENDER_DPI/72, PDF_RENDER_DPI/72)
  ├── pix = page.get_pixmap(matrix=mat, alpha=False)
  ├── img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
  └── return pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
```

**`image_extractor.py`**
```
extract_text_from_image(image_path: str) -> dict
  # Returns: {file_name, text, char_count, preprocessing, error}
  └── _preprocess_image(cv2.imread(image_path)) → preprocessed_img
        ├── cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # grayscale
        ├── if width < 1000: cv2.resize(scale=2.0)   # upscale
        ├── cv2.GaussianBlur((1,1), 0)               # denoise
        └── cv2.adaptiveThreshold(ADAPTIVE_THRESH_GAUSSIAN_C, blockSize=11, C=2)
  └── pytesseract.image_to_string(Image.fromarray(preprocessed), config=TESSERACT_CONFIG)
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
  ├── _discover_files(input_path) → list[Path]  (rglob for all supported extensions)
  ├── for each file_path:
  │     └── _process_single_file(file_path, output_path) → {status, output_path}
  │           ├── guard: 0-byte → status:"empty", skip
  │           ├── extract: pdf → extract_text_from_pdf()
  │           │            image → extract_text_from_image()
  │           │            text → _read_plain_text() (try utf-8, latin-1, cp1252)
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
    "layer": "layer1_document_ingestion"
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
| `src/layer2/trainer.py` | LoRA fine-tuning of ClinicalBERT on BC5CDR — **runs on Colab T4 (Step 34)** |
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
LORA_RANK          = 8
LORA_ALPHA         = 16
LORA_TARGET_MODULES = ["query", "value"]
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

**`entity_linker.py`** — ⚠️ Fixed: uses doc.char_span() not doc[char:char]
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

link_entities(doc: spacy.Doc, entities: list[dict]) -> list[dict]
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
              concept_name = linker.kb.cui_to_entity[concept_id].canonical_name
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
  # Before Step 34 (fine-tuning), this returns a placeholder.
  # After Step 34, loads from models/lora_weights/clinicalbert_rel/
  └── AutoTokenizer.from_pretrained(CLINICALBERT_MODEL)
  └── AutoModelForSequenceClassification.from_pretrained(
          "models/lora_weights/clinicalbert_rel/"  # after fine-tuning
      )

extract_relations(sentences: list[str], entities: list[dict]) -> list[dict]
  # Before fine-tuning: return []  ← placeholder, Layer 3 still builds graphs
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
  │     │     ├── entities  = link_entities(spacy_doc, entities)  # use char_span()
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
  # Key: concept_id (or entity text if None); Value: 768-dim numpy array
  └── for each unique concept_id:
        text = entity["concept_name"] or entity["text"]
        return {concept_id: encode_text(text, tokenizer, model)}
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
  ├── build entity_index: {concept_id → integer_node_idx}
  ├── build visit_index: {doc_filename → integer_visit_idx}
  ├── node_embeddings = encode_entity_nodes(all_entities, tokenizer, encoder_model)
  ├── entity_x = torch.stack([tensor(node_embeddings[e["concept_id"]]) for e in entities])
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
  └── return graph

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
  "visit_dates": ["2023-09-12", "2023-10-05"],
  "graph_file": "mtsamples_0001_graph.pt",
  "source_dataset": "mtsamples"
}
```

---

### Layer 4 — RAG-Augmented Risk Inference  

**Goal:** (1) Build FAISS index from all patient graph embeddings. (2) At inference: retrieve top-5 similar patients, run GraphSAGE, concat RAG context, predict 3 risk scores.

#### Files to create

| File | Role |
|---|---|
| `src/layer4/config.py` | GNN dims, FAISS paths, training hyperparameters |
| `src/layer4/clinical_dataset.py` | PyTorch Geometric Dataset class |
| `src/layer4/faiss_indexer.py` | Build + query FAISS index |
| `src/layer4/rag_retriever.py` | Retrieve similar patients + format context |
| `src/layer4/graph_model.py` | ClinicalGraphSAGE model |
| `src/layer4/risk_heads.py` | MultiTaskRiskModel with 3 risk heads |
| `src/layer4/trainer.py` | Training loop (runs on Colab T4) |
| `src/layer4/pipeline.py` | Batch inference over all graphs |
| `run_layer4.py` | CLI entry point |

#### Function-level breakdown

**`config.py`**
```python
GRAPHSAGE_HIDDEN_DIM = 256
GRAPHSAGE_NUM_LAYERS = 2
FAISS_INDEX_DIM      = 256
FAISS_INDEX_PATH     = "data/corpus_index/faiss.index"
FAISS_IDS_PATH       = "data/corpus_index/patient_ids.json"
FAISS_TOP_K          = 5
RISK_THRESHOLD       = {"readmission": 0.5, "deterioration": 0.6, "medication": 0.5}
LEARNING_RATE        = 2e-4
NUM_EPOCHS           = 20
BATCH_SIZE           = 16
POSITIVE_CLASS_WEIGHT = 3.0   # readmission ~25% of corpus
LABELS_CSV_PATH      = "data/corpus_labels.csv"
# NOTE: No LoRA here — GraphSAGE (~500K params) trained fully end-to-end
# LoRA belongs in Layer 2 (ClinicalBERT) and Layer 5 (BioGPT) only
```

**`clinical_dataset.py`**
```
class ClinicalGraphDataset(InMemoryDataset):
  __init__(self, graphs_folder: str, labels_csv: str):
    ├── load all *_graph.pt files from graphs_folder
    ├── load labels from labels_csv (columns: patient_id, readmission, deterioration, medication)
    ├── for each graph: attach .y_readmission, .y_deterioration, .y_medication as tensors
    └── call self.process()

  __len__(self) -> int
  __getitem__(self, idx: int) -> HeteroData

collate_fn(batch: list[HeteroData]) -> Batch
  # HeteroData CANNOT be stacked by default DataLoader — must use PyG's Batch class
  └── return torch_geometric.data.Batch.from_data_list(batch)
```

**`faiss_indexer.py`**
```
build_index(embeddings: np.ndarray, patient_ids: list[str]) -> None
  └── index = faiss.IndexFlatL2(FAISS_INDEX_DIM)
  └── index.add(embeddings.astype(np.float32))
  └── faiss.write_index(index, FAISS_INDEX_PATH)
  └── json.dump(patient_ids, open(FAISS_IDS_PATH, "w"))

load_index() -> (faiss.Index, list[str])
  └── faiss.read_index(FAISS_INDEX_PATH)
  └── json.load(open(FAISS_IDS_PATH))

query_index(index, query: np.ndarray, top_k=FAISS_TOP_K) -> list[dict]
  └── D, I = index.search(query.reshape(1,-1).astype(np.float32), top_k)
  └── return [{"patient_id": patient_ids[i], "distance": D[0][r], "rank": r+1}
              for r, i in enumerate(I[0])]
```

**`rag_retriever.py`**
```
retrieve_similar_patients(embedding, index, patient_ids, top_k) -> list[dict]
  └── results = query_index(index, embedding, top_k)
  └── for each result: load extracted JSON, extract top-5 entities by link_score
  └── return [{"patient_id", "top_entities": list[dict], "risk_label", "distance"}]

format_rag_context(retrieved: list[dict]) -> torch.Tensor
  # Returns shape [768] — raw mean-pooled entity embeddings
  # The 768→256 projection happens INSIDE MultiTaskRiskModel.forward()
  └── for each retrieved patient: mean-pool their top entity embeddings → [768]
  └── stack K vectors → [K, 768]
  └── mean over K → [768]
  └── return torch.tensor(result, dtype=torch.float32)
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

**`risk_heads.py`**
```
class RiskHead(torch.nn.Module):
  __init__(self, input_dim=512):
    └── Linear(512→128) → ReLU → Dropout(0.3) → Linear(128→1) → Sigmoid

class MultiTaskRiskModel(torch.nn.Module):
  __init__(self, graph_model: ClinicalGraphSAGE):
    └── self.encoder          = graph_model
    └── self.rag_projection   = Linear(768, 256)
        # ⚠️ Fixed: input is 768 (raw entity mean embeddings), NOT 256
        # Projects RAG context 768→256 to match GNN output dim
    └── self.readmission_head  = RiskHead(512)  # 256 GNN + 256 RAG = 512
    └── self.deterioration_head = RiskHead(512)
    └── self.medication_head    = RiskHead(512)

  forward(self, graph: HeteroData, rag_context: torch.Tensor) -> dict:
    └── graph_emb = self.encoder(graph)               # [batch, 256]
    └── rag_emb   = self.rag_projection(rag_context)  # [batch, 768] → [batch, 256]
    └── combined  = torch.cat([graph_emb, rag_emb], dim=-1)  # [batch, 512]
    └── return {
          "readmission":   self.readmission_head(combined),
          "deterioration": self.deterioration_head(combined),
          "medication":    self.medication_head(combined)
        }
```

**`trainer.py`**    ck1
```
# [Colab] — run on Google Colab T4, not local machine
train(config: dict) -> None
  ├── dataset = ClinicalGraphDataset(graphs_folder, LABELS_CSV_PATH)
  ├── train/val/test split = 80/10/10
  ├── DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)
  ├── model = MultiTaskRiskModel(ClinicalGraphSAGE())
  # ⚠️ Do NOT apply LoRA here — GraphSAGE has no "query"/"value" projection layers
  # LoRA only applies to BERT-style attention: ClinicalBERT (L2) and BioGPT (L5)
  ├── optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
  ├── criterion = BCEWithLogitsLoss(pos_weight=tensor(POSITIVE_CLASS_WEIGHT))
  ├── scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
  ├── for epoch in range(NUM_EPOCHS):
  │     ├── train_epoch() → compute 3-head loss, backward, step
  │     ├── evaluate() → AUROC per head (sklearn.metrics.roc_auc_score)
  │     ├── scheduler.step()
  │     └── save checkpoint if val_AUROC improved
  └── log final test metrics

evaluate(model, dataloader) -> dict
  └── collect predictions + labels for all batches
  └── {
        "readmission_auroc":   roc_auc_score(y_true, y_pred),
        "deterioration_auroc": roc_auc_score(...),
        "medication_auroc":    roc_auc_score(...)
      }
```

**`pipeline.py`**
```
run_pipeline(input_folder: str, output_folder: str) -> dict
  # Batch inference over ALL *_graph.pt files in input_folder
  ├── model = load MultiTaskRiskModel from models/graph_model/ + models/risk_heads/
  ├── index, patient_ids = load_index()
  ├── for each *_graph.pt in input_folder:
  │     ├── graph = torch.load(graph_path, weights_only=False)
  │     │   # ⚠️ weights_only=False required: HeteroData is not a plain tensor dict
  │     ├── emb      = get_patient_embedding(graph, model)
  │     ├── retrieved = retrieve_similar_patients(emb, index, patient_ids)
  │     ├── rag_ctx  = format_rag_context(retrieved)  # shape [768]
  │     ├── preds    = model(graph, rag_ctx)
  │     ├── apply RISK_THRESHOLD → binary flags
  │     └── save_json(predictions, f"{output_folder}/{patient_id}_predictions.json")
  └── return {"processed": N, "failed": M, "errors": [...]}
```

#### Expected output schema

```json
{
  "metadata": {
    "patient_id": "mtsamples_0001",
    "layer": "layer4_risk_inference",
    "predicted_at": "2024-01-15T15:00:00",
    "corpus_source": "mtsamples+openI"
  },
  "risk_scores": {
    "readmission_30d":     { "probability": 0.74, "binary": 1, "threshold": 0.5 },
    "acute_deterioration": { "probability": 0.41, "binary": 0, "threshold": 0.6 },
    "medication_risk":     { "probability": 0.62, "binary": 1, "threshold": 0.5 }
  },
  "retrieved_similar_patients": [
    { "patient_id": "mtsamples_0293", "distance": 0.12, "rank": 1 },
    { "patient_id": "openI_1872",     "distance": 0.19, "rank": 2 }
  ],
  "graph_embedding_dim": 256
}
```

---

### Layer 5 — Explainable Output

**Goal:** SHAP feature attribution + BioGPT counterfactual explanation → patient-readable report.

#### Files to create

| File | Role |
|---|---|
| `src/layer5/config.py` | BioGPT model ID, LoRA config, SHAP settings |
| `src/layer5/shap_explainer.py` | SHAP DeepExplainer on risk heads |
| `src/layer5/counterfactual_generator.py` | BioGPT + LoRA explanation generation |
| `src/layer5/finetuner.py` | LoRA fine-tuning of BioGPT on synthetic explanations — **runs on Colab T4 (Step 71)** |
| `src/layer5/report_builder.py` | Assemble final JSON report |
| `src/layer5/pipeline.py` | Orchestrate all XAI steps |
| `run_layer5.py` | CLI entry point |

#### Function-level breakdown

**`config.py`**
```python
from shared.constants import BIOGPT_MODEL

LORA_RANK           = 8
LORA_ALPHA          = 16
LORA_TARGET_MODULES = ["q_proj", "v_proj"]   # BioGPT projection names
EXPLANATION_MAX_TOKENS  = 200
NUM_TOP_SHAP_FEATURES   = 3
PUBMEDQA_DATASET    = "qiaojin/PubMedQA"
PUBMEDQA_CONFIG     = "pqa_labeled"
PUBMEDQA_TRUST_REMOTE = True    # REQUIRED for datasets >= 4.0
MEDMCQA_DATASET     = "openlifescienceai/medmcqa"
```

**`shap_explainer.py`**
```
build_shap_explainer(model: MultiTaskRiskModel, background_embeddings: torch.Tensor) -> shap.DeepExplainer
  # background_embeddings: 50 training-corpus patient embeddings, shape [50, 512]
  # Wraps the risk head portion only (after the 512-dim combined embedding)
  └── risk_head_fn = lambda x: model.readmission_head(x)
  └── shap.DeepExplainer(risk_head_fn, background_embeddings)

compute_shap_values(explainer, patient_embedding: torch.Tensor, risk_type: str) -> np.ndarray
  └── explainer.shap_values(patient_embedding)   # shape [1, 512]
  └── select for risk_type (readmission/deterioration/medication)
  └── return shape [512]

get_top_shap_features(shap_values: np.ndarray, entity_index: dict, n=NUM_TOP_SHAP_FEATURES) -> list[dict]
  └── map shap_values[:256] back to GNN entity embeddings via entity_index
  └── aggregate by entity: mean abs(shap) across embedding dims
  └── sort descending by mean_abs_shap
  └── return top-n as [{"entity_name", "concept_id", "shap_value", "direction"}]
      # ⚠️ field is "concept_id" NOT "cui" — consistent with Layer 2 output
      # direction = "increases_risk" if shap_value > 0, else "decreases_risk"
```

**`counterfactual_generator.py`**
```
load_generator() -> (tokenizer, model)
  └── AutoTokenizer.from_pretrained(BIOGPT_MODEL)
  └── base = AutoModelForCausalLM.from_pretrained(BIOGPT_MODEL)
  └── model = PeftModel.from_pretrained(base, "models/lora_weights/biogpt_explainer/")
  └── model.eval()

build_prompt(patient_id, risk_scores, top_features) -> str
  └── f"Patient risk: readmission={risk_scores['readmission_30d']['probability']:.2f}. "
      f"Top factors: "
      f"1. {top_features[0]['entity_name']} (impact: {top_features[0]['shap_value']:.2f}) "
      f"2. {top_features[1]['entity_name']} (impact: {top_features[1]['shap_value']:.2f}) "
      f"3. {top_features[2]['entity_name']} (impact: {top_features[2]['shap_value']:.2f}). "
      f"Generate a plain English explanation for the patient:"

generate_explanation(prompt, tokenizer, model) -> str
  └── inputs = tokenizer(prompt, return_tensors="pt")
  └── with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=EXPLANATION_MAX_TOKENS, do_sample=False)
  └── full_text = tokenizer.decode(output[0], skip_special_tokens=True)
  └── return full_text[len(prompt):]   # strip prompt prefix
```

**`report_builder.py`**
```
format_risk_level(prob: float) -> str
  └── "low" if prob < 0.3 else "moderate" if prob < 0.6 else "high"

build_report(patient_id, risk_predictions, top_shap_features, explanation_text, similar_patients) -> dict
  └── return {
        "metadata": {...},
        "risk_summary": {
          "readmission_30d":     {"probability": ..., "level": format_risk_level(...)},
          "acute_deterioration": {...},
          "medication_risk":     {...}
        },
        "explanation": {
          "plain_english": explanation_text,
          "top_factors": top_shap_features   # uses concept_id, NOT cui
        },
        "similar_patients_note": "...",
        "disclaimer": "Research prototype — not clinical advice."
      }

save_report(report, output_path) -> None
  └── json.dump(report, open(output_path,"w"), ensure_ascii=False, indent=2)
```

**`pipeline.py`**
```
run_pipeline(predictions_folder, graphs_folder, output_folder) -> dict
  ├── tokenizer, generator = load_generator()
  ├── background_embeddings = load 50 sample embeddings from training corpus
  ├── explainer = build_shap_explainer(model, background_embeddings)
  ├── for each *_predictions.json in predictions_folder:
  │     ├── predictions = load_json(prediction_file)
  │     ├── graph = torch.load(corresponding *_graph.pt, weights_only=False)
  │     ├── meta = load_json(corresponding *_graph_meta.json)
  │     ├── for risk_type in ["readmission","deterioration","medication"]:
  │     │     ├── shap_vals   = compute_shap_values(explainer, embedding, risk_type)
  │     │     └── top_features = get_top_shap_features(shap_vals, meta["entity_index"])
  │     ├── prompt      = build_prompt(patient_id, predictions, top_features)
  │     ├── explanation = generate_explanation(prompt, tokenizer, generator)
  │     ├── report      = build_report(patient_id, predictions, top_features, explanation, retrieved)
  │     └── save_report(report, f"{output_folder}/{patient_id}_report.json")
  └── return summary dict
```

#### Expected output schema

```json
{
  "metadata": { "patient_id": "mtsamples_0001", "layer": "layer5_explainable_output" },
  "risk_summary": {
    "readmission_30d":     { "probability": 0.74, "level": "high" },
    "acute_deterioration": { "probability": 0.41, "level": "moderate" },
    "medication_risk":     { "probability": 0.62, "level": "high" }
  },
  "explanation": {
    "plain_english": "Your readmission risk is elevated. Key factors are rising creatinine levels across visits, hypertension, and frequent recent admissions. If creatinine returned to normal, risk would decrease from 74% to approximately 41%.",
    "top_factors": [
      { "entity_name": "creatinine elevation", "concept_id": "D003404", "shap_value": 0.31, "direction": "increases_risk" },
      { "entity_name": "visit frequency",      "concept_id": null,       "shap_value": 0.22, "direction": "increases_risk" },
      { "entity_name": "hypertension",         "concept_id": "D006973",  "shap_value": 0.19, "direction": "increases_risk" }
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

Step P2  Confirm Tesseract OCR installed:
           tesseract --version
           ✅ VERIFY: output starts with "tesseract 5."
           If not: download from https://github.com/UB-Mannheim/tesseract/wiki

Step P3  Confirm Git installed:
           git --version
           ✅ VERIFY: output starts with "git version"

Step P4  Confirm internet access:
           python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=5); print('OK')"
           ✅ VERIFY: prints "OK"

Step P5  Confirm Google Colab accessible:
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
           mkdir data\raw\pdfs data\raw\images data\raw\txt
           mkdir data\processed data\extracted data\graphs data\predictions data\explanations
           mkdir data\mtsamples data\mtsamples_processed data\mtsamples_extracted data\mtsamples_graphs
           mkdir data\openI data\openI_processed data\openI_extracted data\openI_graphs
           mkdir data\bc5cdr data\ncbi_disease data\corpus_index
           mkdir models\lora_weights\clinicalbert_rel models\lora_weights\biogpt_explainer
           mkdir models\graph_model models\risk_heads
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
           data/corpus_index/
           data/corpus_labels.csv
           data/synthetic_explanations.json
           models/
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
           ✅ VERIFY: pip show transformers torch pdfplumber — all show installed versions
           ⚠️  If bert-score fails: pip install bert-score --no-deps

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
           import pdfplumber, fitz, pytesseract, spacy, scispacy
           import torch, datasets, torch_geometric, faiss, shap, pydantic
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
         BIOGPT_MODEL       = "microsoft/BioGPT-Large"
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

### PHASE 3 — Layer 1 (Weeks 1–2)

```
Step 14  Create src/layer1/config.py  (see Layer 1 function spec above)
Step 15  Create src/layer1/pdf_extractor.py
Step 16  Create src/layer1/image_extractor.py
Step 17  Create src/layer1/text_cleaner.py
Step 18  Create src/layer1/pipeline.py

Step 19  Create run_layer1.py with this EXACT content:
         ────────────────────────────────────────────────────
         import argparse, sys
         from pathlib import Path
         sys.path.insert(0, str(Path(__file__).parent / "src"))
         from layer1.pipeline import run_pipeline

         if __name__ == "__main__":
             parser = argparse.ArgumentParser(description="CHARTA Layer 1 — Document Ingestion")
             parser.add_argument("--input",  default="data/raw",       help="Input folder")
             parser.add_argument("--output", default="data/processed", help="Output folder")
             args = parser.parse_args()
             summary = run_pipeline(args.input, args.output)
             exit(0 if summary["failed"] == 0 else 1)
         ────────────────────────────────────────────────────

Step 20  Copy 3 sample files into tests/sample_data/:
           Download any 3 MTSamples transcriptions as .txt files from mtsamples.com
           OR create synthetic content matching the format in Section 5, Layer 1 output schema
           ✅ VERIFY: dir tests\sample_data — shows 3 .txt files

Step 21  Run tests: pytest tests/test_layer1.py -v
           ✅ VERIFY: All tests PASSED — zero failures before continuing

Step 22  Run pipeline on sample data:
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
Step 23  Create src/layer2/config.py  (see Layer 2 function spec above)
Step 24  Create src/layer2/ner_extractor.py
           ⚠️  extract_entities(doc: spacy.Doc, sentence_idx: int) — NOT (sentences, nlp)
Step 25  Create src/layer2/entity_linker.py
           ⚠️  use doc.char_span(start, end, alignment_mode="expand") — NOT doc[start:end]
Step 26  Create src/layer2/relation_extractor.py  (returns [] placeholder until Step 34)
Step 27  Create src/layer2/temporal_normalizer.py
Step 28  Create src/layer2/pipeline.py
           ⚠️  iterate sentences with enumerate(), call extract_entities(doc, idx)

Step 29  Create run_layer2.py with this template:
         ────────────────────────────────────────────────────
         import argparse, sys
         from pathlib import Path
         sys.path.insert(0, str(Path(__file__).parent / "src"))
         from layer2.pipeline import run_pipeline
         # ⚠️ B16 FIX: do NOT import trainer/evaluator at top level — those modules
         # are only available after Steps 33a–33b; import them inside the elif block.

         if __name__ == "__main__":
             parser = argparse.ArgumentParser(description="CHARTA Layer 2 — NLP Extraction")
             parser.add_argument("--input",   default="data/processed")
             parser.add_argument("--output",  default="data/extracted")
             parser.add_argument("--mode",    default="run",
                                 choices=["run", "finetune", "eval"])
             parser.add_argument("--dataset", default=None)
             args = parser.parse_args()

             if args.mode == "run":
                 run_pipeline(args.input, args.output)
             elif args.mode == "finetune":
                 from layer2.trainer import finetune_relation_model   # lazy import
                 finetune_relation_model(dataset_name=args.dataset or "bigbio/bc5cdr")
             elif args.mode == "eval":
                 from layer2.evaluator import evaluate_ner             # lazy import
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

Step 33a Create src/layer2/trainer.py locally with this content:
         ────────────────────────────────────────────────────
         """
         src/layer2/trainer.py
         Fine-tunes ClinicalBERT for Chemical-Induced Disease (CID) relation extraction
         using LoRA on BC5CDR. Designed to run on Google Colab T4 (Step 34).
         """
         import json
         from pathlib import Path
         from typing import Optional
         from datasets import load_dataset
         from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                                   TrainingArguments, Trainer)
         from peft import LoraConfig, get_peft_model, TaskType
         import torch
         from shared.utils import get_logger
         from layer2.config import (CLINICALBERT_MODEL, BC5CDR_RE_DATASET,
                                    BC5CDR_RE_CONFIG, LORA_RANK, LORA_ALPHA,
                                    LORA_TARGET_MODULES)

         logger = get_logger(__name__)

         def _build_entity_pair_examples(dataset_split) -> list[dict]:
             """Convert BC5CDR KB split to (sentence, e1, e2, label) classification rows."""
             examples = []
             for item in dataset_split:
                 for passage in item.get("passages", []):
                     text = passage.get("text", "")
                     entities = passage.get("entities", [])
                     diseases  = [e for e in entities if e["type"] == "Disease"]
                     chemicals = [e for e in entities if e["type"] == "Chemical"]
                     rels = {(r["arg1_id"], r["arg2_id"]) for r in item.get("relations", [])}
                     for d in diseases:
                         for c in chemicals:
                             label = 1 if (c["id"], d["id"]) in rels or \
                                         (d["id"], c["id"]) in rels else 0
                             inp = (f"[E1] {c['text'][0] if c['text'] else ''} [/E1] "
                                    f"{text} "
                                    f"[E2] {d['text'][0] if d['text'] else ''} [/E2]")
                             examples.append({"text": inp, "label": label})
             return examples

         def finetune_relation_model(
             dataset_name: str = BC5CDR_RE_DATASET,
             config_name: str = BC5CDR_RE_CONFIG,
             output_dir: str = "models/lora_weights/clinicalbert_rel/",
             num_epochs: int = 10,
         ) -> None:
             """Fine-tune ClinicalBERT with LoRA for CID relation extraction."""
             logger.info("Loading BC5CDR dataset ...")
             ds = load_dataset(dataset_name, config_name)

             tokenizer = AutoTokenizer.from_pretrained(CLINICALBERT_MODEL)
             model = AutoModelForSequenceClassification.from_pretrained(
                 CLINICALBERT_MODEL, num_labels=2)

             lora_cfg = LoraConfig(
                 task_type=TaskType.SEQ_CLS,
                 r=LORA_RANK,
                 lora_alpha=LORA_ALPHA,
                 target_modules=LORA_TARGET_MODULES,
                 lora_dropout=0.1,
                 bias="none",
             )
             model = get_peft_model(model, lora_cfg)
             model.print_trainable_parameters()

             def tokenize(batch):
                 return tokenizer(batch["text"], truncation=True,
                                  padding="max_length", max_length=256)

             train_rows = _build_entity_pair_examples(ds["train"])
             val_rows   = _build_entity_pair_examples(ds["validation"])
             from datasets import Dataset
             train_ds = Dataset.from_list(train_rows).map(tokenize, batched=True)
             val_ds   = Dataset.from_list(val_rows).map(tokenize, batched=True)
             train_ds.set_format("torch", columns=["input_ids","attention_mask","label"])
             val_ds.set_format("torch",   columns=["input_ids","attention_mask","label"])

             args = TrainingArguments(
                 output_dir=output_dir,
                 num_train_epochs=num_epochs,
                 per_device_train_batch_size=16,
                 per_device_eval_batch_size=32,
                 evaluation_strategy="epoch",
                 save_strategy="epoch",
                 load_best_model_at_end=True,
                 metric_for_best_model="eval_loss",
                 learning_rate=2e-4,
                 weight_decay=0.01,
                 fp16=True,                  # requires GPU — will error on CPU; OK on Colab T4
                 logging_steps=50,
                 report_to="none",
             )
             trainer = Trainer(model=model, args=args,
                               train_dataset=train_ds, eval_dataset=val_ds)
             logger.info("Starting LoRA fine-tuning of ClinicalBERT ...")
             trainer.train()
             model.save_pretrained(output_dir)
             tokenizer.save_pretrained(output_dir)
             logger.info(f"LoRA weights saved to {output_dir}")
         ────────────────────────────────────────────────────
         ✅ VERIFY (local): python -c "from layer2.trainer import finetune_relation_model; print('trainer OK')"

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

Step 34  [Colab] Fine-tune ClinicalBERT for relation extraction on BC5CDR:

         ── BEFORE YOU OPEN COLAB ────────────────────────────────────────────────────
         You need to prepare a zip archive of the files Colab needs.
         Run on your LOCAL machine (inside the CHARTA/ root):

           # Windows CMD — zip the required folders into one archive
           powershell Compress-Archive -Path src,data\bc5cdr,data\corpus_labels.csv,requirements.txt,conftest.py -DestinationPath charta_colab_l2.zip

         Confirm the zip was created:
           dir charta_colab_l2.zip
         ──────────────────────────────────────────────────────────────────────────────

         ── COLAB NOTEBOOK (paste each block as a separate cell) ──────────────────────

         CELL 1 — Mount Google Drive (recommended — keeps files if session dies):
         ┌─────────────────────────────────────────────────────────────────────────────
         from google.colab import drive
         drive.mount('/content/drive')
         import os
         os.makedirs('/content/drive/MyDrive/CHARTA', exist_ok=True)
         print("Drive mounted at /content/drive/MyDrive/CHARTA")
         └─────────────────────────────────────────────────────────────────────────────

         CELL 2 — Upload zip OR copy from Drive:
         ┌─────────────────────────────────────────────────────────────────────────────
         # Option A: upload the zip file directly (small sessions)
         from google.colab import files
         uploaded = files.upload()          # a dialog opens — select charta_colab_l2.zip

         # Option B: if you already copied the zip to Google Drive manually:
         # !cp "/content/drive/MyDrive/CHARTA/charta_colab_l2.zip" /content/
         └─────────────────────────────────────────────────────────────────────────────

         CELL 3 — Extract zip and set up working directory:
         ┌─────────────────────────────────────────────────────────────────────────────
         import zipfile, os
         with zipfile.ZipFile("charta_colab_l2.zip", "r") as z:
             z.extractall("/content/CHARTA")
         os.chdir("/content/CHARTA")
         import sys
         sys.path.insert(0, "/content/CHARTA/src")
         print("Working dir:", os.getcwd())
         print("Folders:", os.listdir("."))
         └─────────────────────────────────────────────────────────────────────────────

         CELL 4 — Install dependencies (Colab T4 has torch pre-installed — skip torch):
         ┌─────────────────────────────────────────────────────────────────────────────
         # Check existing torch version first
         import torch; print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())

         !pip install -q transformers==4.40.0 peft==0.10.0 accelerate==0.29.3 \
                         datasets==4.0.0 bioc==2.1 scikit-learn==1.4.2 \
                         pydantic==2.7.0 tqdm==4.66.2
         # Note: do NOT reinstall torch on Colab — it breaks CUDA drivers
         └─────────────────────────────────────────────────────────────────────────────

         CELL 5 — Create required output directories:
         ┌─────────────────────────────────────────────────────────────────────────────
         import os
         os.makedirs("models/lora_weights/clinicalbert_rel", exist_ok=True)
         os.makedirs("logs", exist_ok=True)
         print("Output dirs ready")
         └─────────────────────────────────────────────────────────────────────────────

         CELL 6 — Verify GPU is available:
         ┌─────────────────────────────────────────────────────────────────────────────
         import torch
         assert torch.cuda.is_available(), "No GPU! Go to Runtime → Change runtime type → T4 GPU"
         print(f"GPU: {torch.cuda.get_device_name(0)}")
         print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
         └─────────────────────────────────────────────────────────────────────────────

         CELL 7 — Run fine-tuning (~2 hours on T4):
         ┌─────────────────────────────────────────────────────────────────────────────
         from layer2.trainer import finetune_relation_model
         finetune_relation_model(
             dataset_name="bigbio/bc5cdr",
             config_name="bc5cdr_bigbio_kb",
             output_dir="models/lora_weights/clinicalbert_rel/",
             num_epochs=10
         )
         print("Fine-tuning complete!")
         └─────────────────────────────────────────────────────────────────────────────

         CELL 8 — Verify weights were saved:
         ┌─────────────────────────────────────────────────────────────────────────────
         import os
         files_saved = os.listdir("models/lora_weights/clinicalbert_rel/")
         print("Files saved:", files_saved)
         # Expected: ['adapter_config.json', 'adapter_model.safetensors',
         #            'tokenizer_config.json', 'vocab.txt', ...]
         assert "adapter_config.json" in files_saved, "adapter_config.json missing!"
         has_weights = any(f in files_saved for f in
                           ["adapter_model.safetensors", "adapter_model.bin"])
         assert has_weights, "No adapter weights file found!"
         print("Weights verified ✅")
         └─────────────────────────────────────────────────────────────────────────────

         CELL 9 — Zip and download the LoRA weights:
         ┌─────────────────────────────────────────────────────────────────────────────
         import zipfile, os
         zip_path = "/content/clinicalbert_rel_lora.zip"
         with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
             for fname in os.listdir("models/lora_weights/clinicalbert_rel/"):
                 z.write(f"models/lora_weights/clinicalbert_rel/{fname}", fname)
         print(f"Zip created: {os.path.getsize(zip_path) / 1e6:.1f} MB")

         # Option A: download directly to your browser
         from google.colab import files
         files.download(zip_path)

         # Option B: copy to Google Drive for safe storage
         # !cp /content/clinicalbert_rel_lora.zip \
         #     "/content/drive/MyDrive/CHARTA/clinicalbert_rel_lora.zip"
         └─────────────────────────────────────────────────────────────────────────────

         ── AFTER DOWNLOADING ─────────────────────────────────────────────────────────
         Back on your LOCAL machine:
           # Unzip into the correct folder
           powershell Expand-Archive -Path clinicalbert_rel_lora.zip -DestinationPath models\lora_weights\clinicalbert_rel\ -Force
         ──────────────────────────────────────────────────────────────────────────────

         ✅ VERIFY (local, after download):
           python -c "
           import os
           files = os.listdir('models/lora_weights/clinicalbert_rel/')
           print('Files:', files)
           has_weights = any(f in files for f in ['adapter_model.safetensors','adapter_model.bin'])
           assert 'adapter_config.json' in files and has_weights
           print('ClinicalBERT LoRA weights present ✅')
           "
         # Note: PEFT 0.10.0 saves adapter_model.safetensors (not .bin) by default

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
           import pathlib, shutil, argparse

           SECTIONS = ["COMPARISON", "INDICATION", "FINDINGS", "IMPRESSION"]

           def parse_openI_xml_report(xml_path: str) -> dict:
               tree = etree.parse(xml_path)
               uid  = tree.find(".//uId").get("id", pathlib.Path(xml_path).stem)
               result = {"uid": uid}
               for section in SECTIONS:
                   elem = tree.find(f".//AbstractText[@Label='{section}']")
                   result[section.lower()] = (elem.text or "").strip() if elem is not None else ""
               return result

           def write_report_as_text(report: dict, output_dir: str) -> None:
               lines = []
               for section in SECTIONS:
                   text = report.get(section.lower(), "")
                   if text:
                       lines.append(f"{section}:\n{text}")
               pathlib.Path(output_dir, f"openI_{report['uid']}.txt").write_text(
                   "\n\n".join(lines), encoding="utf-8", errors="replace")

           def run(input_dir, txt_output, img_output):
               pathlib.Path(txt_output).mkdir(parents=True, exist_ok=True)
               pathlib.Path(img_output).mkdir(parents=True, exist_ok=True)
               for xml_path in pathlib.Path(input_dir).glob("*.xml"):
                   report = parse_openI_xml_report(str(xml_path))
                   write_report_as_text(report, txt_output)
               for png in pathlib.Path(input_dir).glob("*.png"):
                   shutil.copy(png, img_output)
           ─────────────────────────────────────────────────────────
           python scripts/prepare_openI.py --input data/openI/ --txt_output data/raw/txt/openI/ --img_output data/raw/images/openI/
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

### PHASE 7 — Layer 4: FAISS Index (Week 6)

```
Step 52  Create src/layer4/config.py
Step 53  Create src/layer4/clinical_dataset.py
Step 54  Create src/layer4/faiss_indexer.py

Step 55  Create run_layer4.py with mode support:
         ────────────────────────────────────────────────────
         import argparse, sys
         from pathlib import Path
         sys.path.insert(0, str(Path(__file__).parent / "src"))

         if __name__ == "__main__":
             parser = argparse.ArgumentParser(description="CHARTA Layer 4")
             parser.add_argument("--mode", choices=["build_index","train","eval","run"],
                                 default="run")
             parser.add_argument("--graphs",  nargs="+", default=["data/mtsamples_graphs","data/openI_graphs"])
             parser.add_argument("--output",  default="data/corpus_index")
             parser.add_argument("--input",   default="data/graphs")
             parser.add_argument("--predictions", default="data/predictions")
             args = parser.parse_args()

             if args.mode == "build_index":
                 from layer4.faiss_indexer import build_index_from_folders
                 build_index_from_folders(args.graphs, args.output)
             elif args.mode == "train":
                 from layer4.trainer import train
                 train({})
             elif args.mode == "eval":
                 from layer4.trainer import evaluate_on_test
                 evaluate_on_test()
             elif args.mode == "run":
                 from layer4.pipeline import run_pipeline
                 run_pipeline(args.input, args.predictions)
         ────────────────────────────────────────────────────

Step 56  Build FAISS index from both graph folders:
           python run_layer4.py --mode build_index --graphs data/mtsamples_graphs data/openI_graphs --output data/corpus_index
           ✅ VERIFY: python -c "
           import faiss, json
           idx = faiss.read_index('data/corpus_index/faiss.index')
           ids = json.load(open('data/corpus_index/patient_ids.json'))
           print(f'FAISS index: {idx.ntotal} vectors, {len(ids)} patient IDs')
           assert idx.ntotal > 2000
           print('FAISS index valid ✅')
           "
```

---

### PHASE 8 — Layer 4: GNN Training (Weeks 6–8) [Colab]

```
Step 57  Create src/layer4/rag_retriever.py
Step 58  Create src/layer4/graph_model.py
           ⚠️  self.rag_projection = Linear(768, 256)  NOT Linear(256, 256)
Step 59  Create src/layer4/risk_heads.py
Step 60  Create src/layer4/trainer.py
           ⚠️  Do NOT apply LoRA here — GraphSAGE has no "query"/"value" layers
           ⚠️  Use BCEWithLogitsLoss, NOT BCELoss (includes sigmoid — avoids double sigmoid)
Step 61  Create src/layer4/pipeline.py
           ⚠️  torch.load(path, weights_only=False) — required for HeteroData objects

Step 62  [Colab] Upload training data and run training:
           # In Colab:
           # Mount Google Drive or upload data/mtsamples_graphs/, data/openI_graphs/,
           # data/corpus_labels.csv, data/corpus_index/, and the src/ folder
           !pip install torch_geometric torch peft accelerate faiss-cpu
           !python run_layer4.py --mode train
           # Expected: 2–4 hours on free T4 GPU
           # Download models/graph_model/ and models/risk_heads/ back to local machine
           ✅ VERIFY: dir models\graph_model — shows checkpoint files

Step 63  Run evaluation on held-out test split:
           python run_layer4.py --mode eval
           ✅ VERIFY: AUROC > 0.65 for all three heads (readmission, deterioration, medication)
           → If AUROC < 0.55 (barely above random): check that corpus_labels.csv has correct
             label distribution; verify graph embedding shapes are consistent

Step 64  Run tests: pytest tests/test_layer4.py -v
           ✅ VERIFY: All tests PASSED
```

---

### PHASE 9 — Layer 5 (Weeks 9–10)

```
Step 65  Download PubMedQA (trust_remote_code=True is mandatory):
           python -c "
           from datasets import load_dataset
           ds = load_dataset('qiaojin/PubMedQA', 'pqa_labeled', trust_remote_code=True)
           print('PubMedQA loaded:', len(ds['train']), 'training examples')
           "
           ✅ VERIFY: prints "PubMedQA loaded: 500" or similar (expert-annotated subset)

Step 66  Download MedMCQA:
           python -c "from datasets import load_dataset; ds=load_dataset('openlifescienceai/medmcqa'); print('MedMCQA loaded')"
           ✅ VERIFY: prints "MedMCQA loaded"

Step 67  Create src/layer5/config.py
Step 68  Create src/layer5/shap_explainer.py
           ⚠️  get_top_shap_features returns "concept_id" NOT "cui"
Step 69  Create src/layer5/counterfactual_generator.py

Step 70  Build synthetic_explanations.json (50–100 fine-tuning pairs):
           Create data/synthetic_explanations.json manually with this structure:
           [
             {
               "prompt": "Patient risk: readmission=0.74. Top factors: 1. hypertension (impact: 0.31) 2. creatinine elevation (impact: 0.22) 3. visit frequency (impact: 0.19). Generate plain English explanation:",
               "completion": "Your readmission risk is elevated primarily due to your hypertension and rising creatinine levels across recent visits. If your creatinine returns to normal, your risk would decrease substantially."
             },
             ...
           ]
           Use PubMedQA long_answer fields as templates for medical language style
           ✅ VERIFY: python -c "import json; data=json.load(open('data/synthetic_explanations.json')); print(len(data),'pairs')"
           → must be >= 50

Step 71  [Colab] Fine-tune BioGPT with LoRA:
           !python run_layer5.py --mode finetune \
                                 --data data/synthetic_explanations.json \
                                 --output models/lora_weights/biogpt_explainer/
           Expected: ~1 hour on T4
           Download models/lora_weights/biogpt_explainer/ back to local machine
           ✅ VERIFY: dir models\lora_weights\biogpt_explainer — shows adapter_config.json

Step 72  Create src/layer5/report_builder.py
Step 73  Create src/layer5/pipeline.py

Step 74  Create run_layer5.py:
         ────────────────────────────────────────────────────
         import argparse, sys
         from pathlib import Path
         sys.path.insert(0, str(Path(__file__).parent / "src"))

         if __name__ == "__main__":
             parser = argparse.ArgumentParser(description="CHARTA Layer 5")
             parser.add_argument("--mode",        choices=["run","finetune"], default="run")
             parser.add_argument("--predictions", default="data/predictions")
             parser.add_argument("--graphs",      default="data/graphs")
             parser.add_argument("--output",      default="data/explanations")
             parser.add_argument("--data",        default="data/synthetic_explanations.json")
             args = parser.parse_args()

             if args.mode == "run":
                 from layer5.pipeline import run_pipeline
                 run_pipeline(args.predictions, args.graphs, args.output)
             elif args.mode == "finetune":
                 from layer5.finetuner import finetune_biogpt
                 finetune_biogpt(args.data, "models/lora_weights/biogpt_explainer/")
         ────────────────────────────────────────────────────

Step 75  Run tests: pytest tests/test_layer5.py -v
           ✅ VERIFY: All tests PASSED

Step 76  Run Layer 5:
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

### PHASE 10 — End-to-End Pipeline (Week 10)

```
Step 77  Create run_pipeline.py with this EXACT content:
         ────────────────────────────────────────────────────
         """
         run_pipeline.py — CHARTA full end-to-end pipeline
         Processes a folder of raw medical documents through all 5 layers.
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

             print("[Layer 1] Document Ingestion...")
             l1(input_folder, "data/processed")

             print("[Layer 2] Clinical NLP Extraction...")
             l2("data/processed", "data/extracted")

             print("[Layer 3] Temporal Document Graph...")
             l3("data/extracted", "data/graphs")

             print("[Layer 4] Risk Inference...")
             l4("data/graphs", "data/predictions")

             print("[Layer 5] Explainable Output...")
             l5("data/predictions", "data/graphs", "data/explanations")

             elapsed = round(time.time() - start, 1)
             print(f"\n{'='*60}")
             print(f"Pipeline complete in {elapsed}s")
             print(f"Reports in: data/explanations/")
             print(f"{'='*60}\n")

         if __name__ == "__main__":
             parser = argparse.ArgumentParser(description="CHARTA End-to-End Pipeline")
             parser.add_argument("--input", default="data/raw", help="Folder of raw medical documents")
             args = parser.parse_args()
             run_full_pipeline(args.input)
         ────────────────────────────────────────────────────

Step 78  Test end-to-end pipeline on 3 sample documents:
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

### PHASE 11 — Ablation + Evaluation (Weeks 11–12)

```
Step 79  Ablation A — No RAG context:
           Modify Layer 4 pipeline to pass rag_context = torch.zeros(768)
           Run evaluation; record AUROC for all 3 heads
           ✅ RECORD: save results to results/ablation_table.csv row "no_rag"

Step 80  Ablation B — No temporal edges:
           Modify Layer 3 edge_typer.py to return [] from build_temporal_edges()
           Re-run Layers 3+4; record AUROC
           ✅ RECORD: save to ablation_table.csv row "no_temporal"

Step 81  Ablation C — Attention vs SHAP faithfulness:
           In Layer 5: compute attention weights from last ClinicalBERT layer
           Rank entities by attention weight; compare rank correlation with SHAP ranking
           ✅ RECORD: Pearson r — save to ablation_table.csv row "attention_baseline"

Step 82  Ablation D — Zero-shot vs LoRA BioGPT:
           In Layer 5: load BioGPT WITHOUT LoRA weights
           Generate explanations; compare ROUGE-L and BERTScore vs fine-tuned version
           ✅ RECORD: save to ablation_table.csv row "zero_shot_biogpt"

Step 83  Verify ablation results saved:
           python -c "import pandas as pd; df=pd.read_csv('results/ablation_table.csv'); print(df)"
```

---

### PHASE 12 — Paper and Submission (Weeks 13–16)

```
Step 84  Write paper sections in order:
           Abstract → Introduction → Related Work → Method → Experiments → Results → Conclusion
           ✅ TARGET: ~8 pages for workshop papers, ~12 for full conference

Step 85  Generate all figures:
           Figure 1: CHARTA system architecture (5-layer pipeline diagram)
           Figure 2: AUROC curves for all 3 risk heads
           Figure 3: Ablation study bar chart from results/ablation_table.csv
           Figure 4: Sample patient report with explanation

Step 86  Prepare GitHub repository:
           git init
           git add .
           git commit -m "CHARTA v3.0 — MSc AI Project"
           Push to GitHub (check .gitignore excludes all datasets and model weights)
           Add README.md with: abstract, requirements, dataset download steps, usage examples
           ✅ VERIFY: clone the repo in a fresh folder; verify all dataset download steps work

Step 87  Submit:
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

- **Layer isolation:** layers communicate ONLY via JSON files on disk. No direct cross-layer function calls.
- **Shared code:** anything used by 2+ layers goes in `src/shared/`. Never copy-paste between layers.
- **Model loading:** load models ONCE at the start of a pipeline run. Never inside per-document or per-patient loops.
- **No global mutable state:** never use module-level variables that change at runtime. Pass everything through function arguments.
- **`weights_only=False` on torch.load:** required for HeteroData objects — PyTorch 2.0+ raises FutureWarning without it, will error in 2.4+.

---

## 8. Testing Strategy

### Test coverage requirements

| Test file | What it must test |
|---|---|
| `test_layer1.py` | PDF extraction, OCR extraction, text cleaning (placeholders, abbreviations, segmentation), empty file handling, pipeline batch summary |
| `test_layer2.py` | NER output format, entity linking returns concept_id not cui, char_span alignment, relation extraction threshold filter, temporal ISO normalisation |
| `test_layer3.py` | Graph node count ≥ MIN_ENTITIES, edge types all present, embedding shape [N,768], no NaN values, validate_graph raises on bad input |
| `test_layer4.py` | FAISS build + query, model forward pass output shapes [batch,1] per head, risk scores in [0,1], torch.load with weights_only=False |
| `test_layer5.py` | SHAP values shape matches embedding dim, top_factors use concept_id not cui, explanation string non-empty, report JSON has all required keys |

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
| Relation Extraction (L2) | Micro-F1 on BC5CDR test | > 0.65 | BC5CDR | After Step 34 fine-tuning |
| Readmission Risk (L4) | AUROC on held-out test | > 0.65 | MTSamples + OpenI | Random = 0.50 |
| Deterioration Risk (L4) | AUROC | > 0.65 | MTSamples + OpenI | |
| Medication Risk (L4) | AUROC | > 0.65 | MTSamples + OpenI | |
| Explanation Faithfulness (L5) | Pearson r (SHAP rank vs attention rank) | > 0.60 | MTSamples predictions | |
| Explanation Quality (L5) | BERTScore F1 vs reference | > 0.70 | PubMedQA-derived pairs | |
| Counterfactual Accuracy (L5) | % with correct causal direction | > 0.75 | Manual check (10 samples) | |

---

## 10. Known Bug Registry

All bugs identified across v1.0–v2.4 and fixed in v3.0:

| ID | Severity | Location | Bug | Fix |
|---|---|---|---|---|
| B1 | High | Layer 2 `ner_extractor.py` | `extract_entities` signature took `(sentences, nlp)` but pipeline called with `(doc, idx)` | Signature fixed to `(doc: spacy.Doc, sentence_idx: int)` |
| B2 | High | Layer 2 `entity_linker.py` | `doc[char:char]` used token indices to index by character offset — wrong spans | Replaced with `doc.char_span(start, end, alignment_mode="expand")` |
| B3 | High | Layer 5 JSON output | `"cui"` field used for 2 entities, `"concept_id"` for the third — inconsistent | All entities now use `"concept_id"` exclusively |
| B4 | High | Layer 5 `shap_explainer.py` | `get_top_shap_features` returned `"cui"` key | Changed to `"concept_id"` |
| B5 | High | Tech stack table | `datasets==2.19.0` — too old; `trust_remote_code` not supported until v3.x | Updated to `datasets==4.0.0` |
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

---

*End of CHARTA AI Agent Implementation Guide v3.0*
*All 15 bugs from v1.0–v2.4 have been fixed. Prerequisites section added.
Execution plan restructured for AI agent step-by-step execution with ✅ VERIFY checkpoints at every step.
All runnable code stubs provided for shared utilities, runner scripts, and data preparation scripts.*
