"""Layer 4 — Training loop (runs on Colab T4)."""

from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

from layer4.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    POSITIVE_CLASS_WEIGHT,
    RANDOM_SEED,
    GRAD_CLIP_MAX_NORM,
    LABELS_CSV_PATH,
    DEFAULT_GRAPH_FOLDERS,
    RISK_THRESHOLD,
)
from layer4.clinical_dataset import ClinicalGraphDataset, collate_fn
from layer4.graph_model import ClinicalGraphSAGE
from layer4.readmission_head import ReadmissionRiskModel

logger = logging.getLogger(__name__)


def _set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _split_dataset(
    dataset: ClinicalGraphDataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> tuple[Subset, Subset, Subset]:
    """Split dataset into train/val/test subsets.

    Parameters
    ----------
    dataset : ClinicalGraphDataset
        The full dataset.
    train_ratio : float
        Fraction for training (default: 0.8).
    val_ratio : float
        Fraction for validation (default: 0.1).
    test_ratio : float
        Fraction for testing (default: 0.1).

    Returns
    -------
    tuple[Subset, Subset, Subset]
        Train, validation, and test subsets.
    """
    n = len(dataset)
    if n == 0:
        raise ValueError("Dataset is empty — cannot split")

    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    logger.info(
        "Split: train=%d, val=%d, test=%d (total=%d)",
        len(train_indices), len(val_indices), len(test_indices), n,
    )

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


def train_epoch(
    model: ReadmissionRiskModel,
    dataloader: DataLoader,
    optimizer: AdamW,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch.

    Parameters
    ----------
    model : ReadmissionRiskModel
        The full model (encoder + head).
    dataloader : DataLoader
        Training data loader.
    optimizer : AdamW
        Optimizer.
    criterion : torch.nn.Module
        Loss function (BCEWithLogitsLoss).
    device : torch.device
        CUDA or CPU device.

    Returns
    -------
    float
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # ── Extract batch_dict from PyG Batch ────────────────────────
        # BUG-FIX: PyG Batch._batch_dict is a property that calls
        # collect('_batch'), which raises KeyError when stores lack
        # '_batch'. Build manually from per-node-type .batch vectors.
        batch_dict = {}
        for nt in batch.node_types:
            if hasattr(batch[nt], "batch"):
                batch_dict[nt] = batch[nt].batch
        if not batch_dict:
            batch_dict = None

        # Forward pass — raw logit output (NO sigmoid in model)
        logits = model(batch.x_dict, batch.edge_index_dict, batch_dict)  # [batch_size, 1]

        # ── Extract labels from batched HeteroData ───────────────────
        # y_readmission is attached as a graph-level attribute.
        # After batching, it's stored at batch.y_readmission as a
        # 1D tensor of shape [batch_size] (each graph contributes a scalar).
        labels = batch.y_readmission.to(device)  # [batch_size]

        # BCEWithLogitsLoss expects logits and targets of matching shape
        logits = logits.squeeze(-1)  # [batch_size]

        # Sanity check: catch NaN/Inf in logits or labels
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.warning("NaN/Inf in logits — skipping batch")
            continue

        loss = criterion(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("NaN/Inf loss — skipping batch")
            continue

        loss.backward()

        # Gradient clipping for stability on T4
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def evaluate(
    model: ReadmissionRiskModel,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate model on a dataset.

    Collects predictions + labels for all batches, then computes AUROC.

    Parameters
    ----------
    model : ReadmissionRiskModel
        The full model (encoder + head).
    dataloader : DataLoader
        Evaluation data loader.
    device : torch.device
        CUDA or CPU device.

    Returns
    -------
    dict
        ``{"readmission_auroc": float}`` — AUROC score.
    """
    model.eval()
    all_logits: list[float] = []
    all_labels: list[int] = []

    for batch in dataloader:
        batch = batch.to(device)

        batch_dict = {}
        for nt in batch.node_types:
            if hasattr(batch[nt], "batch"):
                batch_dict[nt] = batch[nt].batch
        if not batch_dict:
            batch_dict = None

        logits = model(batch.x_dict, batch.edge_index_dict, batch_dict)  # [batch_size, 1]
        logits = logits.squeeze(-1)  # [batch_size]

        # Apply sigmoid to convert logits → probabilities for AUROC
        probs = torch.sigmoid(logits)

        all_logits.extend(probs.cpu().tolist())
        all_labels.extend(batch.y_readmission.cpu().tolist())

    # Compute AUROC
    if len(set(all_labels)) < 2:
        logger.warning("Only one class present in labels — AUROC undefined")
        auroc = 0.0
    else:
        auroc = roc_auc_score(all_labels, all_logits)

    return {"readmission_auroc": auroc}


def train(config: dict | None = None) -> None:
    """Train the ClinicalGraphSAGE + ReadmissionHead model.

    # [Colab] — run on Google Colab T4, not local machine

    Parameters
    ----------
    config : dict | None
        Optional config overrides. If None, uses defaults from layer4.config.
    """
    # ── Set seed for reproducibility ─────────────────────────────────
    _set_seed()

    # ── Merge config ─────────────────────────────────────────────────
    if config:
        graphs_folder = config.get("graphs_folder", DEFAULT_GRAPH_FOLDERS)
        labels_csv = config.get("labels_csv", LABELS_CSV_PATH)
        batch_size = config.get("batch_size", BATCH_SIZE)
        learning_rate = config.get("learning_rate", LEARNING_RATE)
        num_epochs = config.get("num_epochs", NUM_EPOCHS)
        positive_class_weight = config.get("positive_class_weight", POSITIVE_CLASS_WEIGHT)
        checkpoint_dir = config.get("checkpoint_dir", "models")
    else:
        graphs_folder = DEFAULT_GRAPH_FOLDERS
        labels_csv = LABELS_CSV_PATH
        batch_size = BATCH_SIZE
        learning_rate = LEARNING_RATE
        num_epochs = NUM_EPOCHS
        positive_class_weight = POSITIVE_CLASS_WEIGHT
        checkpoint_dir = "models"

    # ── Device ───────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Dataset ──────────────────────────────────────────────────────
    dataset = ClinicalGraphDataset(graphs_folder, labels_csv)
    logger.info("Dataset size: %d graphs", len(dataset))

    if len(dataset) == 0:
        logger.error("Dataset is empty — cannot train. Check graph folders and labels CSV.")
        return

    # ── Train/val/test split = 80/10/10 ─────────────────────────────
    train_set, val_set, test_set = _split_dataset(dataset)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ── Model ────────────────────────────────────────────────────────
    graph_encoder = ClinicalGraphSAGE()
    model = ReadmissionRiskModel(graph_encoder).to(device)

    # ⚠️ Do NOT apply LoRA here — GraphSAGE has no "query"/"value" projection layers
    # LoRA belongs in Layer 2 (ClinicalBERT) only

    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # ── Optimizer ────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)

    # ── Loss: BCEWithLogitsLoss with pos_weight ──────────────────────
    # BCEWithLogitsLoss applies sigmoid internally — NO sigmoid in model forward
    pos_weight = torch.tensor(positive_class_weight, device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Scheduler ────────────────────────────────────────────────────
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ── Training loop ────────────────────────────────────────────────
    best_val_auroc = 0.0
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        # ── Train ────────────────────────────────────────────────────
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        logger.info("Epoch %d/%d — train_loss: %.4f", epoch + 1, num_epochs, train_loss)

        # ── Evaluate on validation set ───────────────────────────────
        val_metrics = evaluate(model, val_loader, device)
        val_auroc = val_metrics["readmission_auroc"]
        logger.info("Epoch %d/%d — val_AUROC: %.4f", epoch + 1, num_epochs, val_auroc)

        # ── Scheduler step ───────────────────────────────────────────
        scheduler.step()

        # ── Save checkpoint if val_AUROC improved ───────────────────
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            ckpt_path = checkpoint_dir_path / "best_readmission_model.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auroc": val_auroc,
                "train_loss": train_loss,
            }, str(ckpt_path))
            logger.info("\u2713 Saved checkpoint — val_AUROC improved to %.4f", val_auroc)

    # ── Final test metrics ───────────────────────────────────────────
    test_metrics = evaluate(model, test_loader, device)
    logger.info(
        "Final test — readmission_auroc: %.4f (best_val: %.4f)",
        test_metrics["readmission_auroc"], best_val_auroc,
    )


def evaluate_on_test(
    checkpoint_path: str = "models/best_readmission_model.pt",
    graphs_folder: str | list[str] | None = None,
    labels_csv: str | None = None,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Load the best checkpoint and evaluate on the test split.

    Called by ``run_layer4.py --mode eval``.

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved best model checkpoint.
    graphs_folder : str | list[str] | None
        Path to folder(s) with ``*_graph.pt`` files.
        Defaults to DEFAULT_GRAPH_FOLDERS.
    labels_csv : str | None
        Path to ``corpus_labels.csv``.
        Defaults to LABELS_CSV_PATH.
    batch_size : int
        Batch size for evaluation.

    Returns
    -------
    dict
        Test metrics with ``readmission_auroc`` key.
    """
    _set_seed()

    if graphs_folder is None:
        graphs_folder = DEFAULT_GRAPH_FOLDERS
    if labels_csv is None:
        labels_csv = LABELS_CSV_PATH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluating on device: %s", device)

    # ── Dataset + split ──────────────────────────────────────────────
    dataset = ClinicalGraphDataset(graphs_folder, labels_csv)

    if len(dataset) == 0:
        logger.error("Dataset is empty — cannot evaluate.")
        return {"readmission_auroc": 0.0}

    _, _, test_set = _split_dataset(dataset)

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ── Load model from checkpoint ───────────────────────────────────
    graph_encoder = ClinicalGraphSAGE()
    model = ReadmissionRiskModel(graph_encoder)

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    logger.info(
        "Loaded checkpoint from %s (epoch %d, val_AUROC %.4f)",
        checkpoint_path, checkpoint.get("epoch", -1), checkpoint.get("val_auroc", 0.0),
    )

    # ── Evaluate ─────────────────────────────────────────────────────
    test_metrics = evaluate(model, test_loader, device)
    logger.info("Test — readmission_auroc: %.4f", test_metrics["readmission_auroc"])
    return test_metrics