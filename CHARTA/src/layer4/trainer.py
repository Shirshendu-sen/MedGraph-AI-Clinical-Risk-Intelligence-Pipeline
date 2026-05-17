import os
import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_geometric.loader import DataLoader

from layer4.config import (
    LABELS_CSV_PATH,
    LEARNING_RATE,
    NUM_EPOCHS,
    BATCH_SIZE,
    FAISS_TOP_K,
    POSITIVE_CLASS_WEIGHT
)

from layer4.clinical_dataset import ClinicalGraphDataset
from layer4.graph_model import ClinicalGraphSAGE
from layer4.risk_heads import MultiTaskRiskModel

from layer4.faiss_indexer import load_index
from layer4.rag_retriever import (
    retrieve_similar_patients,
    format_rag_context
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def compute_auroc(y_true, y_pred):

    try:
        return roc_auc_score(y_true, y_pred)
    except:
        return 0.5


def evaluate(
    model,
    dataloader,
    index,
    patient_ids,
    criterion
):

    model.eval()

    total_loss = 0

    y_true_read = []
    y_pred_read = []

    y_true_det = []
    y_pred_det = []

    y_true_med = []
    y_pred_med = []

    with torch.no_grad():

        for batch in dataloader:

            batch = batch.to(DEVICE)

            # -------------------------------
            # Graph embeddings
            # -------------------------------

            graph_emb = model.encoder(
                batch.x_dict,
                batch.edge_index_dict,
                batch.batch_dict
            )

            # -------------------------------
            # RAG retrieval
            # -------------------------------

            rag_contexts = []

            for emb in graph_emb:

                retrieved = retrieve_similar_patients(
                    emb.detach().cpu().numpy(),
                    index,
                    patient_ids,
                    top_k=FAISS_TOP_K
                )

                rag_ctx = format_rag_context(retrieved)

                rag_contexts.append(rag_ctx)

            rag_contexts = torch.stack(
                rag_contexts
            ).to(DEVICE)

            # -------------------------------
            # Forward pass
            # -------------------------------

            preds = model(batch, rag_contexts)

            y = batch.y.float()

            loss_read = criterion(
                preds["readmission"].squeeze(),
                y[:, 0]
            )

            loss_det = criterion(
                preds["deterioration"].squeeze(),
                y[:, 1]
            )

            loss_med = criterion(
                preds["medication"].squeeze(),
                y[:, 2]
            )

            loss = (
                loss_read +
                loss_det +
                loss_med
            )

            total_loss += loss.item()

            # -------------------------------
            # Store predictions
            # -------------------------------

            read_probs = torch.sigmoid(
                preds["readmission"]
            ).squeeze()

            det_probs = torch.sigmoid(
                preds["deterioration"]
            ).squeeze()

            med_probs = torch.sigmoid(
                preds["medication"]
            ).squeeze()

            y_true_read.extend(
                y[:, 0].cpu().numpy()
            )

            y_pred_read.extend(
                read_probs.cpu().numpy()
            )

            y_true_det.extend(
                y[:, 1].cpu().numpy()
            )

            y_pred_det.extend(
                det_probs.cpu().numpy()
            )

            y_true_med.extend(
                y[:, 2].cpu().numpy()
            )

            y_pred_med.extend(
                med_probs.cpu().numpy()
            )

    metrics = {

        "loss": total_loss / len(dataloader),

        "readmission_auroc":
            compute_auroc(
                y_true_read,
                y_pred_read
            ),

        "deterioration_auroc":
            compute_auroc(
                y_true_det,
                y_pred_det
            ),

        "medication_auroc":
            compute_auroc(
                y_true_med,
                y_pred_med
            )
    }

    metrics["mean_auroc"] = np.mean([
        metrics["readmission_auroc"],
        metrics["deterioration_auroc"],
        metrics["medication_auroc"]
    ])

    return metrics


def train(config):

    # -----------------------------------
    # Dataset
    # -----------------------------------

    dataset = ClinicalGraphDataset(
        graphs_folder=config["graphs_folder"],
        labels_csv=LABELS_CSV_PATH
    )

    # -----------------------------------
    # Split
    # -----------------------------------

    # Index-based split (sklearn train_test_split cannot handle HeteroData objects)
    indices = np.arange(len(dataset))
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    # -----------------------------------
    # Loaders
    # -----------------------------------

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE
    )

    test_loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE
    )

    # -----------------------------------
    # Metadata
    # -----------------------------------

    metadata = dataset[0].metadata()

    # -----------------------------------
    # Models
    # -----------------------------------

    encoder = ClinicalGraphSAGE(
        metadata=metadata
    )

    model = MultiTaskRiskModel(
        graph_model=encoder
    ).to(DEVICE)

    # -----------------------------------
    # Optimizer
    # -----------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    # -----------------------------------
    # Loss
    # -----------------------------------

    pos_weight = torch.tensor([POSITIVE_CLASS_WEIGHT]).to(DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # -----------------------------------
    # Scheduler
    # -----------------------------------

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS
    )

    # -----------------------------------
    # FAISS
    # -----------------------------------

    index, patient_ids = load_index()

    # -----------------------------------
    # Checkpoints
    # -----------------------------------

    os.makedirs(
        "models/risk_heads",
        exist_ok=True
    )

    best_score = 0

    # -----------------------------------
    # Training loop
    # -----------------------------------

    for epoch in range(NUM_EPOCHS):

        model.train()

        total_loss = 0

        for batch in train_loader:

            batch = batch.to(DEVICE)

            optimizer.zero_grad()

            # ---------------------------
            # Graph embeddings
            # ---------------------------

            graph_emb = model.encoder(
                batch.x_dict,
                batch.edge_index_dict,
                batch.batch_dict
            )

            # ---------------------------
            # Build RAG contexts
            # ---------------------------

            rag_contexts = []

            for emb in graph_emb:

                retrieved = retrieve_similar_patients(
                    emb.detach().cpu().numpy(),
                    index,
                    patient_ids,
                    top_k=FAISS_TOP_K
                )

                rag_ctx = format_rag_context(
                    retrieved
                )

                rag_contexts.append(rag_ctx)

            rag_contexts = torch.stack(
                rag_contexts
            ).to(DEVICE)

            # ---------------------------
            # Forward
            # ---------------------------

            preds = model(
                batch,
                rag_contexts
            )

            y = batch.y.float()

            loss_read = criterion(
                preds["readmission"].squeeze(),
                y[:, 0]
            )

            loss_det = criterion(
                preds["deterioration"].squeeze(),
                y[:, 1]
            )

            loss_med = criterion(
                preds["medication"].squeeze(),
                y[:, 2]
            )

            loss = (
                loss_read +
                loss_det +
                loss_med
            )

            # ---------------------------
            # Backward
            # ---------------------------

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        # -----------------------------------
        # Validation
        # -----------------------------------

        metrics = evaluate(
            model,
            val_loader,
            index,
            patient_ids,
            criterion
        )

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS}"
        )

        print(
            f"Train Loss: "
            f"{total_loss / len(train_loader):.4f}"
        )

        print(metrics)

        # -----------------------------------
        # Save best model
        # -----------------------------------

        if metrics["mean_auroc"] > best_score:

            best_score = metrics["mean_auroc"]

            torch.save(

                model.state_dict(),

                "models/risk_heads/best_model.pt"
            )

            print(
                "Best model saved."
            )

    # -----------------------------------
    # Final test evaluation
    # -----------------------------------

    print("\nRunning final test evaluation...\n")

    test_metrics = evaluate(
        model,
        test_loader,
        index,
        patient_ids,
        criterion
    )

    print(test_metrics)


def evaluate_on_test():

    print(
        "Load trained model "
        "and evaluate here."
    )