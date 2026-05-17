"""Layer 4 — RAG-Augmented Risk Inference: PyTorch Geometric Dataset class."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import torch
from torch_geometric.data import HeteroData, Batch

from layer4.config import LABELS_CSV_PATH

logger = logging.getLogger(__name__)


class ClinicalGraphDataset:
    """Dataset of patient heterogeneous graphs with multi-task risk labels.

    Loads ``*_graph.pt`` files produced by Layer 3 and attaches binary
    risk labels (readmission, deterioration, medication) from a CSV file.

    Parameters
    ----------
    graphs_folder : str
        Path to folder containing ``*_graph.pt`` files from Layer 3.
    labels_csv : str
        Path to CSV with columns: patient_id, readmission, deterioration, medication.
    """

    def __init__(
        self,
        graphs_folder: str,
        labels_csv: str = LABELS_CSV_PATH,
    ):
        self.graphs_folder = graphs_folder
        self.labels_csv = labels_csv

        # Load graph files
        self.graph_files = sorted(Path(graphs_folder).glob("*_graph.pt"))
        logger.info("Found %d graph files in %s", len(self.graph_files), graphs_folder)

        # Load labels CSV → dict[patient_id → (readmission, deterioration, medication)]
        self.labels_map: dict[str, tuple[int, int, int]] = {}
        self._load_labels(labels_csv)

        # Load and label each graph
        self.graphs: list[HeteroData] = []
        self._load_and_label_graphs()

    def _load_labels(self, labels_csv: str) -> None:
        """Load risk labels from CSV into a lookup dict."""
        csv_path = Path(labels_csv)
        if not csv_path.exists():
            logger.warning("Labels CSV not found at %s — all labels default to 0", labels_csv)
            return

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row["patient_id"]
                readmission = int(row.get("readmission", 0))
                deterioration = int(row.get("deterioration", 0))
                medication = int(row.get("medication", 0))
                self.labels_map[pid] = (readmission, deterioration, medication)

        logger.info("Loaded labels for %d patients from %s", len(self.labels_map), labels_csv)

    def _load_and_label_graphs(self) -> None:
        """Load each graph PT file and attach risk labels as tensor attributes."""
        for gf in self.graph_files:
            graph = torch.load(str(gf), weights_only=False)

            # Derive patient_id from filename or from graph metadata
            patient_id = getattr(graph["patient"], "patient_id", None)
            if patient_id is None:
                # Fallback: derive from filename stem
                stem = gf.stem  # e.g. "sample_cardiology_graph"
                if stem.endswith("_graph"):
                    patient_id = stem[:-len("_graph")]
                else:
                    patient_id = stem

            # Look up labels; default to (0, 0, 0) if patient not in CSV
            labels = self.labels_map.get(patient_id, (0, 0, 0))

            # Attach labels as tensor attributes on the graph
            graph.y_readmission = torch.tensor([labels[0]], dtype=torch.float32)
            graph.y_deterioration = torch.tensor([labels[1]], dtype=torch.float32)
            graph.y_medication = torch.tensor([labels[2]], dtype=torch.float32)

            # Stacked .y tensor for trainer: [3] — (readmission, deterioration, medication)
            graph.y = torch.tensor([labels[0], labels[1], labels[2]], dtype=torch.float32)

            # Store patient_id for FAISS indexing later
            graph.patient_id = patient_id

            self.graphs.append(graph)

        logger.info("Loaded and labeled %d patient graphs", len(self.graphs))

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> HeteroData:
        return self.graphs[idx]


def collate_fn(batch: list[HeteroData]) -> Batch:
    """Collate a list of HeteroData graphs into a PyG Batch object.

    HeteroData CANNOT be stacked by default DataLoader — must use
    PyG's Batch class.

    Parameters
    ----------
    batch : list[HeteroData]
        List of heterogeneous graph objects.

    Returns
    -------
    Batch
        Batched heterogeneous graph.
    """
    return Batch.from_data_list(batch)