"""Layer 4 — GNN dims, training hyperparameters."""

# ── GraphSAGE architecture ──────────────────────────────────────────
GRAPHSAGE_HIDDEN_DIM  = 256
GRAPHSAGE_NUM_LAYERS  = 2
GRAPHSAGE_IN_DIM      = 768       # matches ClinicalBERT [CLS] embedding dim
GRAPHSAGE_OUT_DIM     = 256       # same as hidden — 2-layer SAGE ends at this dim
GRAPHSAGE_DROPOUT     = 0.3

# ── Readmission head ────────────────────────────────────────────────
RISK_THRESHOLD        = 0.5       # readmission: >= 0.5 → HIGH
READMISSION_HEAD_DIM  = 128       # intermediate layer in ReadmissionHead

# ── Training ────────────────────────────────────────────────────────
LEARNING_RATE         = 2e-4
WEIGHT_DECAY          = 0.01      # AdamW weight decay (was hardcoded in trainer)
NUM_EPOCHS            = 20
BATCH_SIZE            = 16
POSITIVE_CLASS_WEIGHT = 3.0       # readmission ~25% of corpus → up-weight positives
RANDOM_SEED           = 42        # deterministic train/val/test splits
GRAD_CLIP_MAX_NORM    = 1.0       # gradient clipping for Colab stability

# ── Paths ───────────────────────────────────────────────────────────
LABELS_CSV_PATH       = "data/corpus_labels.csv"
DEFAULT_GRAPH_FOLDERS = ["data/mtsamples_graphs", "data/openI_graphs"]

# NOTE: No LoRA here — GraphSAGE (~500K params) trained fully end-to-end.
# LoRA belongs in Layer 2 (ClinicalBERT) only.