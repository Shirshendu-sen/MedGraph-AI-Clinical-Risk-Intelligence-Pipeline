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