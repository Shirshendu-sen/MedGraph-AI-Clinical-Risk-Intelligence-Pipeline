# Layer 2 — Clinical NLP Extraction: Configuration

"""Model names, linker names, thresholds for Layer 2."""

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
