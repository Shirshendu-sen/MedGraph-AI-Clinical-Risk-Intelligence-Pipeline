# MedGraph‑AI‑Clinical‑Risk‑Intelligence‑Pipeline

![License](https://img.shields.io/badge/License-MIT-blue.svg) ![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python) ![AI](https://img.shields.io/badge/AI‑Research‑orange) ![Build](https://img.shields.io/github/actions/workflow/status/yourrepo/ci.yml?branch=main&label=CI)

**A modular, graph‑based AI pipeline that ingests heterogeneous clinical documents, builds temporal knowledge graphs, and delivers explainable risk predictions.**

---

## Project Overview

- **Goal:** Transform raw clinical texts (EHR notes, radiology reports, pathology summaries) into a temporally aware knowledge graph that powers risk stratification and counterfactual explanations.
- **End‑to‑end flow:** OCR → NLP extraction → Entity linking → Temporal graph construction → Retrieval‑augmented generation (RAG) + risk model → Explainability engine → Structured report.

---

## Core Architecture

| Layer | Function | Key Components |
|-------|----------|----------------|
| **Layer 1** | Document ingestion & preprocessing | OCR (`image_extractor.py`), PDF parsing (`pdf_extractor.py`), text cleaning (`text_cleaner.py`) |
| **Layer 2** | Clinical NLP | Named‑entity recognition (`ner_extractor.py`), relation extraction (`relation_extractor.py`), temporal normalisation (`temporal_normalizer.py`), MeSH/RxNorm linking (`entity_linker.py`) |
| **Layer 3** | Temporal Knowledge Graph | Node encoding (`node_encoder.py`), edge typing (`edge_typer.py`), graph builder (`graph_builder.py`) |
| **Layer 4** | RAG + Risk Prediction | Retrieval via FAISS, ClinicalBERT/BioGPT embeddings, risk classifier (Temporal GraphSAGE) |
| **Layer 5** | Explainability Engine | SHAP for feature importance, counterfactual generation, patient‑similarity retrieval |

---

## System Workflow

```
Raw Docs (PDF/TXT) → OCR / PDF Extractor → NLP (NER, Relations) → Entity Linking → Temporal KG Builder → FAISS Retrieval + GraphSAGE Risk Engine → SHAP / Counterfactual Explainability → Clinical Report
```

---

## AI & ML Pipeline

- **ClinicalBERT** – contextual embeddings for clinical entities.
- **BioGPT** – generative augmentation for RAG.
- **GraphSAGE** – temporal graph representation learning for risk scoring.
- **FAISS** – dense vector similarity search for patient‑wise retrieval.
- **SHAP** – model‑agnostic interpretability.
- **RAG** – combines retrieved graph context with generative LLM for report synthesis.

---

## Feature Highlights

- OCR‑driven document ingestion.
- Fine‑grained clinical entity extraction.
- MeSH & RxNorm entity linking.
- Temporal heterogeneous graph construction.
- Risk prediction over patient trajectories.
- Counterfactual and SHAP‑based explanations.
- Patient similarity search via FAISS.

---

## Dataset & Research Stack

| Domain | Datasets |
|--------|----------|
| Clinical Text | **MTSamples**, **OpenI** |
| Entity Recognition | **BC5CDR**, **NCBI Disease** |
| Question Answering | **PubMedQA**, **MedMCQA** |

---

## Technology Stack

| Category | Tools |
|----------|-------|
| **NLP** | spaCy, ClinicalBERT, BioGPT |
| **Graph Learning** | PyTorch‑Geometric, GraphSAGE |
| **Deep Learning** | PyTorch, Transformers |
| **Explainability** | SHAP, Captum |
| **Data Processing** | pandas, FAISS, OCR (Tesseract) |

---

## Folder Structure

```
CHARTA/
├─ data/
│  ├─ raw/                # Original PDFs/TXTs (MTSamples, OpenI)
│  ├─ extracted/          # OCR & PDF text outputs
│  ├─ processed/          # Cleaned, tokenised documents
│  └─ graphs/             # Temporal KG JSONs
├─ src/
│  ├─ layer1/             # Ingestion pipeline
│  ├─ layer2/             # Clinical NLP pipeline
│  ├─ layer3/             # Graph construction
│  ├─ layer4/             # RAG & risk model (placeholder)
│  ├─ layer5/             # Explainability utilities
│  └─ shared/             # Constants, schemas, utils
├─ scripts/                # Data preparation & label generation
├─ tests/                  # Unit tests for each layer
└─ requirements.txt        # Python dependencies
```

---

## Research & Engineering Highlights

- **Modular, layer‑wise design** enables independent development and scaling of each component.
- **Scalable pipeline:** parallel OCR, batched graph construction, and GPU‑accelerated GNN training.
- **Heterogeneous temporal graph modeling** captures disease progression and treatment timelines.
- **Explainable AI:** SHAP + counterfactuals provide clinicians with actionable insights.
- **End‑to‑end clinical reasoning workflow** from raw notes to risk‑aware reports.

---

## Future Improvements

- Multimodal integration (imaging, lab results).
- Advanced graph reasoning (neural symbolic inference).
- Real‑time inference service with asynchronous streaming.
- Distributed GNN training for massive EHR cohorts.
- Full EHR integration (FHIR, HL7) for production deployment.

---

## Authors

**[Your Name]** – AI Research Engineer
**[Collaborator]** – Clinical Informatics Specialist

*This repository reflects a research‑grade implementation of a CHARTA‑style clinical AI pipeline, ready for further development and academic evaluation.*