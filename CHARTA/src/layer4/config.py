"""Layer 4 — RAG-Augmented Risk Inference: GNN dims, FAISS paths, training hyperparameters."""

from shared.constants import CLINICALBERT_MODEL  # do NOT re-define here

# ── GraphSAGE architecture ──
GRAPHSAGE_HIDDEN_DIM    = 256
GRAPHSAGE_NUM_LAYERS    = 2
GRAPHSAGE_INPUT_DIM     = 768   # matches ClinicalBERT [CLS] embedding dim

# ── FAISS index ──
FAISS_INDEX_DIM         = 256   # must match GRAPHSAGE_HIDDEN_DIM
FAISS_INDEX_PATH        = "data/corpus_index/faiss.index"
FAISS_IDS_PATH          = "data/corpus_index/patient_ids.json"
FAISS_TOP_K             = 5

# ── Risk thresholds ──
RISK_THRESHOLD          = {"readmission": 0.5, "deterioration": 0.6, "medication": 0.5}

# ── Training hyperparameters ──
LEARNING_RATE           = 2e-4
NUM_EPOCHS              = 20
BATCH_SIZE              = 16
POSITIVE_CLASS_WEIGHT   = 3.0   # readmission ~25% of corpus

# ── Data paths ──
LABELS_CSV_PATH         = "data/corpus_labels.csv"

# NOTE: No LoRA here — GraphSAGE (~500K params) trained fully end-to-end
# LoRA belongs in Layer 2 (ClinicalBERT) and Layer 5 (BioGPT) only