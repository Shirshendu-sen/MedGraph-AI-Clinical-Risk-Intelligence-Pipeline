<div align="center">

# 🧠 MedGraph-AI — Clinical Risk Intelligence Pipeline

### *Clinical History-Aware RAG-Augmented Temporal Architecture*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗_Transformers-4.40.0-FFD21F?style=for-the-badge)](https://huggingface.co/)
[![PyG](https://img.shields.io/badge/PyG-2.5.2-3C4A8A?style=for-the-badge&logo=graphql&logoColor=white)](https://pyg.org/)
[![FAISS](https://img.shields.io/badge/FAISS-1.8.0-00C4B4?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![SHAP](https://img.shields.io/badge/SHAP-0.45.0-FF6B35?style=for-the-badge)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Research-8B5CF6?style=for-the-badge)](#)
[![Status](https://img.shields.io/badge/Status-MSc_Final_Project-10B981?style=for-the-badge)](#)

<br/>

> **CHARTA** (**C**linical **H**istory-**A**ware **R**AG-augmented **T**emporal **A**rchitecture) is a research-grade, end-to-end clinical AI pipeline that ingests raw medical documents, constructs patient-level temporal knowledge graphs, and produces explainable multi-task risk predictions — with zero gated datasets.

<br/>

```
Raw Docs → [L1 Ingest] → [L2 NLP] → [L3 Graph] → [L4 Risk] → [L5 Explain] → Patient Report
```

</div>

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Core Architecture](#-core-architecture)
- [System Workflow](#-system-workflow)
- [AI & ML Pipeline](#-ai--ml-pipeline)
- [Feature Highlights](#-feature-highlights)
- [Dataset & Research Stack](#-dataset--research-stack)
- [Technology Stack](#-technology-stack)
- [Folder Structure](#-folder-structure)
- [Research & Engineering Highlights](#-research--engineering-highlights)
- [Evaluation Metrics](#-evaluation-metrics)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🔬 Project Overview

CHARTA is a **five-layer modular clinical AI system** designed for temporal clinical risk intelligence. It processes any folder of unstructured medical documents — PDFs, scanned images, plain text — and outputs a structured, explainable patient risk report without requiring access to any gated clinical database.

**Clinical AI Objective:**
Predict three clinically-relevant risk outcomes per patient — **30-day readmission**, **acute deterioration**, and **medication risk** — by reasoning over the patient's full longitudinal clinical history encoded as a heterogeneous temporal knowledge graph.

**End-to-end pipeline:**

| Input | Processing | Output |
|---|---|---|
| Unstructured medical documents (PDF/image/text) | Five sequential AI layers | Risk score + plain-language explanation + patient summary JSON |

**Key design decisions:**
- **Layer isolation via JSON:** each layer communicates only through files on disk — no cross-layer function coupling
- **Zero gated data:** all datasets load anonymously from HuggingFace or public mirrors
- **Ablation-ready:** modular architecture supports ablation studies (no RAG, no temporal edges, attention vs SHAP)

---

## 🏛️ Core Architecture

CHARTA is structured as five isolated, sequentially-dependent layers:

### Layer 1 — Document Ingestion

Transforms heterogeneous raw medical documents into clean, structured text JSON.

- **PDF extraction:** native text via `pdfplumber`; scanned pages rendered to images via `PyMuPDF`
- **OCR pipeline:** `Tesseract 5` + `OpenCV` preprocessing (deskew, threshold, denoise)
- **Text cleaning:** encoding correction (`ftfy`), placeholder normalisation, clinical abbreviation expansion, sentence segmentation
- **Handles:** native-text PDFs, scanned PDFs, radiology images (OpenI), plain-text discharge summaries

### Layer 2 — Clinical NLP Extraction

Reads Layer 1 JSON and produces structured clinical entities with concept IDs, relations, and ISO timestamps.

- **NER:** `scispaCy` (`en_ner_bc5cdr_md`) — extracts `DISEASE` and `CHEMICAL` entity spans
- **Entity linking:** scispaCy built-in `EntityLinker` maps entities to **MeSH** (~30k disease concepts) and **RxNorm** (~100k drug concepts) — zero NIH registration required (replaced archived MedCAT)
- **Relation extraction:** `ClinicalBERT` fine-tuned with **LoRA** (rank 8) on BC5CDR for Chemical-Induced Disease (CID) relation detection
- **Temporal normalisation:** regex + `python-dateutil` → all dates resolved to ISO 8601 (`YYYY-MM-DD`)

### Layer 3 — Temporal Knowledge Graph

Transforms Layer 2 JSON into a patient-level **heterogeneous PyTorch Geometric graph** that encodes clinical history across multiple visits.

**Graph schema:**

```
Nodes:  entity [N_entities, 768]  |  visit [N_visits, 768]  |  patient [1, 768]
Edges:  (entity, occurs_in, visit)        ← membership
        (visit, before, visit)            ← temporal ordering
        (entity, relates_to, entity)      ← CID relations from Layer 2
        (entity, co_occurs_with, entity)  ← sentence co-occurrence (window=3)
```

- **Node embeddings:** ClinicalBERT `[CLS]` token embeddings (768-dim) per entity
- **Visit nodes:** mean-pooled entity embeddings per clinical visit
- **Patient node:** global mean-pool across all visits

### Layer 4 — RAG-Augmented Risk Inference

Two-phase layer: (1) build a FAISS corpus index; (2) at inference, retrieve similar patients and predict three risk scores.

- **GraphSAGE encoder:** `ClinicalGraphSAGE` (2-layer, hidden=256, out=256) over the heterogeneous entity graph
- **FAISS retrieval:** `IndexFlatL2` over 256-dim patient embeddings; top-5 similar patients retrieved per query
- **RAG context fusion:** retrieved patient entity embeddings mean-pooled → [768] → projected [768→256] and concatenated with GNN output [256] → combined [512]
- **Multi-task risk heads:** three independent `RiskHead` modules (512→128→1→Sigmoid) for readmission, deterioration, and medication risk

### Layer 5 — Explainability Engine

Translates raw risk scores into clinically interpretable, plain-language explanations.

- **SHAP attribution:** `shap.DeepExplainer` on the risk head module — maps importance back to named clinical entities via the entity index
- **Counterfactual generation:** `BioGPT` fine-tuned with **LoRA** (rank 8, `q_proj`/`v_proj`) on PubMedQA — generates "what if" explanations in natural language
- **Report assembly:** structured JSON report combining risk scores, top SHAP features, counterfactual explanation, and similar patient references

---

## ⚡ System Workflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CHARTA PIPELINE FLOW                             │
└─────────────────────────────────────────────────────────────────────────┘

  📄 Raw Documents
  (PDF / Image / TXT)
        │
        ▼
┌───────────────────┐
│   LAYER 1         │  pdfplumber · PyMuPDF · Tesseract OCR · OpenCV
│   Document        │  ──────────────────────────────────────────────
│   Ingestion       │  → cleaned_text + sentences + metadata
└────────┬──────────┘
         │  {*_processed.json}
         ▼
┌───────────────────┐
│   LAYER 2         │  scispaCy NER · MeSH/RxNorm Linker
│   Clinical NLP    │  ClinicalBERT + LoRA · python-dateutil
│   Extraction      │  ──────────────────────────────────────
└────────┬──────────┘  → entities + relations + timestamps
         │  {*_extracted.json}
         ▼
┌───────────────────┐
│   LAYER 3         │  ClinicalBERT Encoder · PyTorch Geometric
│   Temporal        │  HeteroData · Temporal + Relation Edges
│   Knowledge Graph │  ──────────────────────────────────────
└────────┬──────────┘  → {patient_id}_graph.pt  [HeteroData]
         │
         ▼
┌───────────────────┐
│   LAYER 4         │  GraphSAGE · FAISS · RAG Retrieval
│   Risk Inference  │  Multi-Task Risk Heads · BCEWithLogitsLoss
│   (GNN + RAG)     │  ──────────────────────────────────────
└────────┬──────────┘  → readmission / deterioration / medication scores
         │  {*_predictions.json}
         ▼
┌───────────────────┐
│   LAYER 5         │  SHAP DeepExplainer · BioGPT + LoRA
│   Explainability  │  Counterfactual Generator · Report Builder
│   Engine          │  ──────────────────────────────────────
└────────┬──────────┘  → plain-language explanation + entity attribution
         │
         ▼
  📊 Patient Report
  {patient_id}_report.json
  risk scores · top factors · counterfactual · similar patients
```

---

## 🤖 AI & ML Pipeline

| Component | Model | Role | Training |
|---|---|---|---|
| **Clinical NER** | `en_ner_bc5cdr_md` (scispaCy) | Biomedical entity span detection | Pre-trained on BC5CDR |
| **Entity Linker** | scispaCy EntityLinker | MeSH + RxNorm concept grounding | Zero-shot (KB lookup) |
| **Relation Extraction** | `ClinicalBERT` + LoRA | Chemical-Induced Disease detection | Fine-tuned on BC5CDR-KB |
| **Node Encoder** | `ClinicalBERT` | 768-dim entity/visit/patient embeddings | Frozen inference |
| **Graph Encoder** | `ClinicalGraphSAGE` | Heterogeneous graph reasoning | Trained end-to-end on Colab T4 |
| **Similarity Retrieval** | `FAISS IndexFlatL2` | Patient-level nearest-neighbour search | Index built from corpus |
| **RAG Context Fusion** | Linear projection (768→256) | Augment GNN with retrieved patient context | Part of multi-task model |
| **Risk Prediction** | `MultiTaskRiskModel` (3 heads) | Readmission · Deterioration · Medication | BCEWithLogitsLoss, AdamW |
| **SHAP Attribution** | `shap.DeepExplainer` | Feature importance → named entities | Post-hoc, no retraining |
| **Explanation Generation** | `BioGPT` + LoRA | Natural-language counterfactual explanation | Fine-tuned on PubMedQA |

**Training configuration:**

```python
GRAPHSAGE_HIDDEN_DIM = 256     LEARNING_RATE  = 2e-4
GRAPHSAGE_NUM_LAYERS = 2       NUM_EPOCHS     = 20
FAISS_TOP_K          = 5       BATCH_SIZE     = 16
POSITIVE_CLASS_WEIGHT = 3.0    LORA_RANK      = 8
```

---

## ✨ Feature Highlights

**📥 Document Ingestion**
- Multi-format ingestion: native-text PDFs, scanned PDFs (OCR), radiology images (OpenI), plain text
- Preprocessing chain: deskew → threshold → denoise → Tesseract → `ftfy` encoding fix → sentence segmentation

**🔬 Clinical Entity Extraction**
- `DISEASE` and `CHEMICAL` span detection using BC5CDR-trained scispaCy model
- Character-offset-correct entity spans (`doc.char_span()` with `alignment_mode="expand"`)

**🔗 MeSH / RxNorm Concept Linking**
- Zero-registration concept grounding via scispaCy's built-in `EntityLinker`
- MeSH for diseases (~30k concepts) · RxNorm for drugs (~100k concepts)
- Top-1 KB candidate + link score stored per entity

**⏱️ Temporal Graph Construction**
- Entities timestamped to ISO 8601 dates extracted from clinical text
- Visit nodes ordered by earliest entity timestamp → `(visit, before, visit)` temporal edges
- Four edge types capture clinical knowledge at different granularities

**📈 Multi-Task Risk Prediction**
- Simultaneous prediction of three clinically distinct risk scores in a single forward pass
- RAG-augmented: similar patient context fused with graph embedding before risk heads
- Per-risk configurable thresholds: `readmission=0.5`, `deterioration=0.6`, `medication=0.5`

**🔍 SHAP Feature Attribution**
- `DeepExplainer` over the 512-dim combined embedding identifies which clinical entities drive each risk score
- Entity-level attribution: SHAP values mapped back to concept names and MeSH/RxNorm IDs
- Direction labelled: `increases_risk` / `decreases_risk` per factor

**💬 Counterfactual Explanations**
- BioGPT (LoRA fine-tuned on PubMedQA) generates "what-if" natural-language explanations
- Prompt constructed from top SHAP features + risk probability values
- ROUGE-L and BERTScore evaluated against PubMedQA-derived reference pairs

**👥 Patient Similarity Retrieval**
- FAISS L2 index over 256-dim GraphSAGE embeddings for the full patient corpus
- Top-5 similar patients retrieved at inference; their entity profiles inform RAG context

---

## 📚 Dataset & Research Stack

| Dataset | Role in CHARTA | Size | Access |
|---|---|---|---|
| **MTSamples** | Primary clinical corpus — Layers 1, 3, 4 training | ~10 MB · 4,999 transcriptions · 40 specialties | ✅ Public GitHub mirror |
| **BC5CDR** (NER) | NER evaluation benchmark — Layer 2 | ~5 MB | ✅ `tner/bc5cdr` on HuggingFace |
| **BC5CDR** (KB) | CID relation extraction fine-tuning — Layer 2 | ~8 MB | ✅ `bigbio/bc5cdr` config `bc5cdr_bigbio_kb` |
| **NCBI Disease** | NER F1 benchmark — Layer 2 evaluation | ~2 MB · 793 abstracts | ✅ `ncbi/ncbi_disease` |
| **OpenI IU-XRay** | Radiology reports + OCR test images — Layers 1, 3 | ~1.5 GB | ✅ `ykumards/open-i` |
| **PubMedQA** | BioGPT fine-tuning for explanation generation — Layer 5 | ~300 MB | ✅ `qiaojin/PubMedQA` (`trust_remote_code=True`) |
| **MedMCQA** | Factuality evaluation — Layer 5 | ~700 MB | ✅ `openlifescienceai/medmcqa` |

> **Zero gated datasets.** No PhysioNet/MIMIC access, no NIH/UMLS registration required. All datasets load anonymously.

**Risk label derivation** (from MTSamples + OpenI metadata via `scripts/generate_labels.py`):

| Label | HIGH-risk condition |
|---|---|
| `readmission` | Cardiovascular / Nephrology / Emergency / Critical Care specialty; or keywords: *readmit, return to ED* |
| `deterioration` | Keywords: *acute, urgent, severe, critical, worsening, ICU transfer, emergent* |
| `medication` | ≥ 5 distinct drug names; or keywords: *drug interaction, polypharmacy, adverse reaction* |

---

## 🛠️ Technology Stack

### NLP & Language Models

| Library | Version | Purpose |
|---|---|---|
| `scispacy` | 0.5.4 | Biomedical NER + MeSH/RxNorm entity linking |
| `transformers` | 4.40.0 | ClinicalBERT · BioGPT loading & inference |
| `peft` | 0.10.0 | LoRA fine-tuning for ClinicalBERT (L2) and BioGPT (L5) |
| `accelerate` | 0.29.3 | PEFT-compatible training acceleration |

### Graph Learning

| Library | Version | Purpose |
|---|---|---|
| `torch_geometric` | 2.5.2 | GraphSAGE · HeteroData heterogeneous graph |
| `torch` | 2.2.2 | Core tensor operations, training loop |
| `faiss-cpu` | 1.8.0 | Patient similarity FAISS index (L2) |

### Document Ingestion

| Library | Version | Purpose |
|---|---|---|
| `pdfplumber` | 0.10.3 | Native-text PDF extraction |
| `PyMuPDF` | 1.24.0 | Scanned PDF → image rendering (no Poppler needed) |
| `pytesseract` | 0.3.10 | Tesseract OCR Python wrapper |
| `opencv-python` | 4.9.0.80 | Image preprocessing for OCR |
| `ftfy` | 6.1.3 | Unicode / encoding correction |
| `lxml` | 5.2.1 | NLM-format XML parsing (OpenI) |

### Explainability & Evaluation

| Library | Version | Purpose |
|---|---|---|
| `shap` | 0.45.0 | SHAP DeepExplainer on risk heads |
| `rouge-score` | 0.1.2 | ROUGE-L for explanation quality |
| `bert-score` | 0.3.13 | BERTScore F1 for explanation quality |
| `scikit-learn` | 1.4.2 | AUROC · F1 · train/val/test splits |

### Data & Utilities

| Library | Version | Purpose |
|---|---|---|
| `datasets` | 4.0.0 | HuggingFace dataset loading (v4 required for `trust_remote_code`) |
| `pydantic` | 2.7.0 | Schema validation for pipeline JSON |
| `python-dateutil` | 2.9.0 | ISO 8601 temporal normalisation |
| `numpy` | 1.26.4 | Array ops |
| `pandas` | 2.2.2 | CSV handling (labels, ablation results) |
| `pytest` | 8.0.0 | Unit + integration testing |

---

## 📁 Folder Structure

```
CHARTA/
│
├── run_pipeline.py              # End-to-end orchestrator
├── run_layer{1-5}.py            # Per-layer CLI entry points
├── requirements.txt
│
├── data/
│   ├── raw/                     # Input: pdfs/ · images/ · txt/
│   ├── processed/               # Layer 1 output
│   ├── extracted/               # Layer 2 output
│   ├── graphs/                  # Layer 3 output (.pt + meta JSON)
│   ├── predictions/             # Layer 4 output
│   ├── explanations/            # Layer 5 output (final reports)
│   ├── mtsamples/               # MTSamples corpus
│   ├── openI/                   # OpenI radiology corpus
│   ├── corpus_index/            # FAISS index + patient_ids.json
│   └── corpus_labels.csv        # readmission / deterioration / medication labels
│
├── models/
│   ├── lora_weights/
│   │   ├── clinicalbert_rel/    # LoRA-tuned ClinicalBERT (L2 relation extraction)
│   │   └── biogpt_explainer/    # LoRA-tuned BioGPT (L5 explanation generation)
│   ├── graph_model/             # Trained ClinicalGraphSAGE weights
│   └── risk_heads/              # Trained MultiTaskRiskModel weights
│
├── src/
│   ├── layer1/                  # pdf_extractor · image_extractor · text_cleaner
│   ├── layer2/                  # ner_extractor · entity_linker · relation_extractor · temporal_normalizer
│   ├── layer3/                  # graph_builder · node_encoder · edge_typer
│   ├── layer4/                  # clinical_dataset · faiss_indexer · rag_retriever · graph_model · risk_heads · trainer
│   ├── layer5/                  # shap_explainer · counterfactual_generator · report_builder
│   └── shared/                  # constants · schema · utils · logger
│
├── scripts/
│   ├── prepare_mtsamples.py
│   ├── prepare_openI.py
│   └── generate_labels.py
│
├── tests/
│   ├── sample_data/             # sample_discharge_summary.txt · sample_lab_report.txt
│   ├── test_layer{1-5}.py
│
└── results/
    └── ablation_table.csv
```

---

## 🔬 Research & Engineering Highlights

**Modular layer-isolated architecture**
Each layer reads JSON from disk and writes JSON to disk. Zero direct cross-layer function calls. This enforces clean boundaries, enables independent testing, and makes ablation studies trivial — swap any layer's output and re-run downstream.

**Heterogeneous temporal graph modelling**
Rather than treating a patient's visits as independent documents, CHARTA constructs a unified multi-relational graph that preserves temporal ordering (`before` edges), clinical co-occurrences, KB-grounded relations, and entity-visit membership simultaneously — enabling the GNN to reason across a patient's longitudinal history.

**RAG-augmented clinical reasoning**
FAISS retrieval injects population-level clinical context — similar patients' entity profiles — into every risk prediction. The 768-dim RAG context is projected and concatenated with the 256-dim GNN embedding, giving the model access to both individual graph structure and corpus-level similarity signals.

**Explainable AI via SHAP + counterfactuals**
Every risk score is accompanied by (a) per-entity SHAP attribution mapping importance back to named MeSH/RxNorm concepts, and (b) a BioGPT-generated natural-language counterfactual explanation — making clinical AI reasoning auditable and interpretable.

**Production-grade code discipline**
Type hints on every function, one-line minimum docstrings, `try/except` on all I/O, structured `{"status", "error", ...}` return dicts, `get_logger(__name__)` throughout, config-file-only constants, and ≥1 pytest test per logic function.

**Ablation study design**
Four ablations defined in the execution plan: (A) no RAG context, (B) no temporal edges, (C) attention weights vs SHAP faithfulness, (D) zero-shot vs LoRA BioGPT — all results logged to `results/ablation_table.csv`.

---

## 📊 Evaluation Metrics

| Component | Metric | Target | Benchmark Dataset |
|---|---|---|---|
| NER | F1-score | > 0.75 | NCBI Disease test set |
| Entity Linking | Accuracy@1 | > 0.70 | MTSamples manual spot-check (n=20) |
| Relation Extraction | Micro-F1 | > 0.65 | BC5CDR test set |
| Readmission Risk | AUROC | > 0.65 | MTSamples + OpenI held-out test |
| Deterioration Risk | AUROC | > 0.65 | MTSamples + OpenI held-out test |
| Medication Risk | AUROC | > 0.65 | MTSamples + OpenI held-out test |
| Explanation Faithfulness | Pearson r (SHAP vs attention rank) | > 0.60 | MTSamples predictions |
| Explanation Quality | BERTScore F1 | > 0.70 | PubMedQA-derived pairs |
| Counterfactual Accuracy | % correct causal direction | > 0.75 | Manual review (n=10) |

---

## 🚀 Future Improvements

**Multimodal clinical AI**
Extend Layer 1 to ingest structured EHR time-series data (vitals, lab trends) alongside unstructured text, and fuse modalities at the graph level via cross-modal edges.

**Advanced graph reasoning**
Replace GraphSAGE with a heterogeneous graph transformer (HGT) to learn attention weights across different edge types, enabling the model to weight temporal vs relational vs co-occurrence signals adaptively.

**Real-time inference**
Package Layers 4–5 as a REST API endpoint with pre-loaded FAISS index and model weights, enabling sub-second risk scoring for individual documents without re-running the full pipeline.

**Distributed graph learning**
For larger corpora, migrate Layer 3 graph construction and Layer 4 training to a distributed PyG framework (e.g. PyG-DistSAGE) to scale beyond single-machine RAM limits.

**Multimodal EHR integration**
Incorporate structured medication lists, ICD code sequences, and lab value time-series directly as graph node features, reducing reliance on NLP extraction for structured clinical data.

---

## 👤 Author

<div align="center">

**CHARTA v3.0** — MSc Final Year Project in Clinical AI

*Developed as a research-grade, submission-ready system targeting ACL BioNLP, EMNLP Clinical NLP, and IEEE JBHI.*

---

*All 15 bugs from v1.0–v2.4 resolved. Zero gated datasets. Full test suite. Ablation-ready.*

</div>