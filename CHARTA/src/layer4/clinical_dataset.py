"""Layer 4 — PyTorch Geometric Dataset class for temporal patient graphs."""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

import torch
from torch_geometric.data import HeteroData, InMemoryDataset, Batch

logger = logging.getLogger(__name__)

# ── Sentinel / fallback tokens ────────────────────────────────────────
_UNK_TOKEN = "UNK"          # standard ML "unknown" sentinel
_UNKNOWN_STRING = "UNKNOWN"  # appears in entity_index from Layer 3 graphs

# ── Non-tensor metadata keys that must be stripped before PyG collation ──
# PyG's HeteroData.collate() iterates ALL stored keys.  If we leave
# dicts, lists, or strings on the object, collate() will try to index
# them via internal key mappings and crash with KeyError (e.g. "UNKNOWN").
_NON_TENSOR_META_KEYS = frozenset({"entity_index", "visit_dates"})

# patient["patient_id"] is a string and also crashes PyG collation
_PATIENT_META_ATTRS = frozenset({"patient_id"})


def _strip_graph_metadata(graph: HeteroData) -> HeteroData:
    """Remove non-tensor metadata from a HeteroData graph so PyG collation
    does not crash on dict keys like ``"UNKNOWN"`` or string attributes.

    This function is idempotent and safe to call multiple times.
    It *copies* the graph to avoid mutating the original.

    Parameters
    ----------
    graph : HeteroData
        Raw graph from Layer 3 with ``entity_index``, ``visit_dates``,
        and ``patient.patient_id`` metadata attached.

    Returns
    -------
    HeteroData
        Clean graph with only tensor attributes preserved for collation.
    """
    # Work on a shallow copy so the original is not mutated
    import copy
    clean = copy.copy(graph)

    # Remove top-level non-tensor attributes
    for key in _NON_TENSOR_META_KEYS:
        if hasattr(clean, key):
            delattr(clean, key)

    # Remove patient-level string attributes
    if "patient" in clean.node_types:
        for attr in _PATIENT_META_ATTRS:
            if hasattr(clean["patient"], attr):
                delattr(clean["patient"], attr)

    return clean


def _is_openi_unlabeled(
    graph: HeteroData,
    graph_path: Path,
    labels_lookup: dict[str, int],
) -> bool:
    """Determine whether an openI graph should be excluded from supervised
    training because it has no matching row in ``corpus_labels.csv``.

    Returns ``True`` if the graph is from the openI corpus AND its
    normalized patient_id is not present in the labels lookup.

    Parameters
    ----------
    graph : HeteroData
        The loaded patient graph.
    graph_path : Path
        Path to the ``*_graph.pt`` file (used for folder-name fallback).
    labels_lookup : dict[str, int]
        The normalized-patient-id → label dictionary.

    Returns
    -------
    bool
        ``True`` if this openI graph has no label and should be skipped.
    """
    # 1. Determine source
    source = "unknown"
    try:
        raw_pid = graph["patient"].patient_id
    except (AttributeError, KeyError):
        raw_pid = ""
    if not raw_pid:
        folder_name = graph_path.parent.name.lower()
        if "openi" in folder_name:
            source = "openI"
        elif "mtsamples" in folder_name:
            source = "mtsamples"
    elif raw_pid.lower().startswith("openi_"):
        source = "openI"
    elif raw_pid.lower().startswith("mtsamples_"):
        source = "mtsamples"

    # 2. Only check openI — mtsamples and sample graphs are always kept
    if source != "openI":
        return False

    # 3. Check if label exists
    normalized_id = _normalize_patient_id(raw_pid) if raw_pid else ""
    if normalized_id and normalized_id in labels_lookup:
        return False  # has a label — keep it

    return True  # openI without label — exclude


def _normalize_patient_id(raw_id: str) -> str:
    """Normalize a patient_id to canonical form for cross-reference matching.

    Graph files produced by Layer 3 carry patient_ids derived from raw
    filenames (e.g. ``mtsamples_0000__allergic_rhinitis_``), while
    corpus_labels.csv stores clean base IDs (e.g. ``mtsamples_0000``).

    This function strips whitespace, trailing underscores, and — for
    ``mtsamples_`` prefixed IDs — extracts only the ``mtsamples_NNNN``
    base portion.  All other IDs (sample, openI, etc.) are returned
    stripped, which preserves compatibility with non-mtsamples corpora.

    Parameters
    ----------
    raw_id : str
        Raw patient_id from the graph or CSV.

    Returns
    -------
    str
        Canonical patient_id suitable for dictionary lookup.
    """
    if not raw_id:
        return ""
    # Strip all whitespace (including non-breaking, tabs, newlines)
    # and trailing underscores from slug-suffixed Layer 3 IDs
    pid = raw_id.strip().rstrip("_")
    if not pid:
        return ""
    # mtsamples IDs carry a slug suffix from the original .txt filename
    # (e.g. "mtsamples_0000__allergic_rhinitis_" → "mtsamples_0000").
    # Extract only the numeric base: mtsamples_NNNN
    match = re.match(r"(mtsamples_\d+)", pid)
    if match:
        return match.group(1)
    # openI IDs (openi_NNNN) and sample IDs (sample_cardiology, etc.)
    # are already clean — return as-is after strip/rstrip
    return pid


class ClinicalGraphDataset(InMemoryDataset):
    """Loads temporal patient history graphs from Layer 3 into PyG Data objects.

    Each ``*_graph.pt`` file produced by Layer 3 is a ``HeteroData`` object
    with node types ``entity``, ``visit``, ``patient`` and edge types
    ``occurs_in``, ``before``, ``relates_to``, ``co_occurs_with``.

    The dataset attaches a ``.y_readmission`` label tensor to each graph
    by looking up the patient_id in ``corpus_labels.csv``.

    Supports loading from multiple graph folders (e.g., separate
    mtsamples_graphs/ and openI_graphs/ directories).
    """

    def __init__(
        self,
        graphs_folder: str | list[str],
        labels_csv: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        """Initialise the dataset.

        Parameters
        ----------
        graphs_folder : str | list[str]
            Path (or list of paths) to folders containing ``*_graph.pt``
            files from Layer 3.
        labels_csv : str
            Path to ``corpus_labels.csv`` with columns:
            patient_id, readmission, deterioration, medication.
        transform : callable, optional
            PyG transform applied on-the-fly.
        pre_transform : callable, optional
            PyG pre-transform applied once during processing.
        pre_filter : callable, optional
            PyG pre-filter applied once during processing.
        """
        # Normalize to list of Paths
        if isinstance(graphs_folder, (str, Path)):
            self._graph_folders = [Path(graphs_folder)]
        else:
            self._graph_folders = [Path(p) for p in graphs_folder]

        self.labels_csv = Path(labels_csv)

        # Load labels lookup: {normalized_patient_id: int}
        self._labels_lookup: dict[str, int] = {}
        self._load_labels()

        # Discover graph files across all folders
        self._graph_files: list[Path] = []
        for folder in self._graph_folders:
            if folder.is_dir():
                found = sorted(folder.glob("*_graph.pt"))
                self._graph_files.extend(found)
                logger.debug("Found %d graph(s) in %s", len(found), folder)
            else:
                logger.warning("Graph folder not found: %s", folder)

        if not self._graph_files:
            logger.warning("No *_graph.pt files found in any graph folder")

        # Use first folder as root for processed/ storage
        root = str(self._graph_folders[0]) if self._graph_folders else "data/graphs"

        super().__init__(
            root=root,
            transform=transform,
            pre_transform=pre_transform,
            pre_filter=pre_filter,
        )

        # Load processed data if it exists, otherwise trigger process()
        if len(self._graph_files) > 0:
            processed_path = Path(self.processed_paths[0])
            if processed_path.exists():
                try:
                    self.data, self.slices = torch.load(
                        str(processed_path), weights_only=False,
                    )
                    logger.debug("Loaded pre-processed data from %s", processed_path)
                except Exception as e:
                    logger.warning(
                        "Failed to load processed cache %s: %s — will re-process",
                        processed_path, e,
                    )
                    processed_path.unlink(missing_ok=True)
                    # Defer processing to first access (lazy via len/get)
            else:
                logger.debug("No processed cache found — will process on first access")

    @property
    def raw_file_names(self) -> list[str]:
        """Not used — we bypass the raw/processed pipeline."""
        return []

    @property
    def processed_file_names(self) -> list[str]:
        """Name of the processed data file."""
        return ["data.pt"]

    def download(self) -> None:
        """No download needed — graphs come from Layer 3."""
        pass

    # ── Label loading ────────────────────────────────────────────────

    def _load_labels(self) -> None:
        """Load readmission labels from CSV into a lookup dict.

        Both CSV keys and future graph lookups are normalized via
        :func:`_normalize_patient_id` so that graph IDs with filename
        slug suffixes (e.g. ``mtsamples_0000_allergic_rhinitis_``)
        match the clean base IDs in the CSV (e.g. ``mtsamples_0000``).

        Tracks per-source label counts (mtsamples, openI, other) so that
        mismatch warnings in :meth:`process` are properly contextualised.
        """
        if not self.labels_csv.exists():
            logger.warning("Labels CSV not found: %s", self.labels_csv)
            return

        skipped_empty = 0
        skipped_duplicate = 0
        source_counts: dict[str, int] = {"mtsamples": 0, "openI": 0, "other": 0}

        with open(self.labels_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_pid = row.get("patient_id", "")
                if not raw_pid or not raw_pid.strip():
                    skipped_empty += 1
                    continue
                pid = _normalize_patient_id(raw_pid)
                if not pid:
                    skipped_empty += 1
                    continue
                if pid in self._labels_lookup:
                    skipped_duplicate += 1
                    logger.debug("Duplicate CSV key after normalization: '%s' (raw: '%s')",
                                 pid, raw_pid.strip())
                    continue
                try:
                    label = int(row.get("readmission", "0"))
                except (ValueError, KeyError):
                    label = 0
                self._labels_lookup[pid] = label

                # Track per-source label counts
                if pid.startswith("mtsamples_"):
                    source_counts["mtsamples"] += 1
                elif pid.startswith("openi_"):
                    source_counts["openI"] += 1
                else:
                    source_counts["other"] += 1

        logger.info(
            "Loaded %d patient labels from %s (mtsamples: %d, openI: %d, other: %d; "
            "skipped: %d empty, %d duplicate)",
            len(self._labels_lookup), self.labels_csv,
            source_counts["mtsamples"], source_counts["openI"], source_counts["other"],
            skipped_empty, skipped_duplicate,
        )
        self._label_source_counts = source_counts

    # ── InMemoryDataset overrides ────────────────────────────────────

    def _get_label_for_graph(self, graph: HeteroData) -> int:
        """Look up the readmission label for a patient graph.

        Normalizes the graph's ``patient_id`` via :func:`_normalize_patient_id`
        before dictionary lookup so that filename-derived IDs (e.g.
        ``mtsamples_0000_allergic_rhinitis_``) match the clean CSV keys
        (e.g. ``mtsamples_0000``).

        Falls back to 0 (no readmission) if patient_id is not in labels CSV
        or the graph is missing a patient_id entirely (corrupted graph).
        """
        try:
            raw_patient_id = graph["patient"].patient_id
        except (AttributeError, KeyError):
            logger.warning(
                "GRAPH ID: <MISSING> | CSV MATCH: N/A | LABEL: NOT FOUND "
                "(graph has no patient_id — corrupted; assigning label 0)"
            )
            return 0

        normalized_id = _normalize_patient_id(raw_patient_id)

        if not normalized_id:
            logger.warning(
                "GRAPH ID: '%s' → normalized: <EMPTY> | CSV MATCH: NO | "
                "LABEL: NOT FOUND (normalized to empty string; defaulting to 0)",
                raw_patient_id,
            )
            return 0

        if normalized_id in self._labels_lookup:
            label = self._labels_lookup[normalized_id]
            logger.debug(
                "GRAPH ID: '%s' → normalized: '%s' | CSV MATCH: YES | LABEL: FOUND (%d)",
                raw_patient_id, normalized_id, label,
            )
            return label

        # Not found — log with full debug context including source hint
        source_hint = ""
        if normalized_id.startswith("mtsamples_"):
            source_hint = " [source: mtsamples]"
        elif normalized_id.startswith("openi_"):
            source_hint = " [source: openI — CSV may not contain openI labels]"
        logger.warning(
            "GRAPH ID: '%s' → normalized: '%s' | CSV MATCH: NO | LABEL: NOT FOUND "
            "(defaulting to 0)%s",
            raw_patient_id, normalized_id, source_hint,
        )
        return 0

    def process(self) -> None:
        """Load all graph files, attach labels, filter invalid graphs,
        and store as InMemoryDataset.

        Called once on first access; results cached to processed_paths[0].
        Tracks per-source label match statistics and validates the final
        class distribution to catch degenerate datasets early.
        """
        import os

        data_list: list[HeteroData] = []
        labels_found = 0
        labels_missing = 0

        # Per-source tracking: {source: {"found": int, "missing": int, "total": int}}
        source_stats: dict[str, dict[str, int]] = {
            "mtsamples": {"found": 0, "missing": 0, "total": 0},
            "openI":    {"found": 0, "missing": 0, "total": 0},
            "unknown":  {"found": 0, "missing": 0, "total": 0},
        }

        # --- stale cache detection ---
        cached_path = Path(self.processed_paths[0])
        if cached_path.exists():
            cache_mtime = os.path.getmtime(str(cached_path))
            stale = False
            for gf in self._graph_files:
                if os.path.getmtime(str(gf)) > cache_mtime:
                    stale = True
                    break
            if not stale and self.labels_csv.exists():
                if os.path.getmtime(str(self.labels_csv)) > cache_mtime:
                    stale = True
            if stale:
                logger.warning(
                    "Stale processed cache detected (%s is older than source files); "
                    "removing to force reprocessing",
                    cached_path.name,
                )
                cached_path.unlink(missing_ok=True)
            else:
                logger.debug("Processed cache is up-to-date: %s", cached_path)

        # --- main loading loop ---
        for graph_path in self._graph_files:
            try:
                graph = torch.load(str(graph_path), weights_only=False)
            except Exception as e:
                logger.warning("Failed to load %s — skipping: %s", graph_path.name, e)
                continue

            # Attach readmission label
            label = self._get_label_for_graph(graph)

            # ── BUG-FIX #1: Exclude openI graphs with no CSV label ────
            # corpus_labels.csv contains only mtsamples_* entries (0 openi_*).
            # Including unlabeled openI graphs would silently assign label 0
            # and pollute supervised training with incorrect ground-truth.
            if _is_openi_unlabeled(graph, graph_path, self._labels_lookup):
                logger.info(
                    "Skipping openI graph %s — no label in corpus_labels.csv",
                    graph_path.name,
                )
                # Track the skip in source stats (as missing)
                source_stats["openI"]["missing"] += 1
                source_stats["openI"]["total"] += 1
                labels_missing += 1
                continue

            graph.y_readmission = torch.tensor(label, dtype=torch.float32)

            # Determine source bucket from filename or patient_id
            source = "unknown"
            try:
                raw_pid = graph["patient"].patient_id
            except (AttributeError, KeyError):
                raw_pid = ""
            if not raw_pid:
                # Fallback: infer from parent folder name
                folder_name = graph_path.parent.name.lower()
                if "openi" in folder_name:
                    source = "openI"
                elif "mtsamples" in folder_name:
                    source = "mtsamples"
            elif raw_pid.lower().startswith("mtsamples_"):
                source = "mtsamples"
            elif raw_pid.lower().startswith("openi_"):
                source = "openI"

            # Track match statistics
            try:
                normalized_id = _normalize_patient_id(raw_pid) if raw_pid else ""
                if normalized_id and normalized_id in self._labels_lookup:
                    labels_found += 1
                    source_stats[source]["found"] += 1
                else:
                    labels_missing += 1
                    source_stats[source]["missing"] += 1
            except (AttributeError, KeyError):
                labels_missing += 1
                source_stats[source]["missing"] += 1
            source_stats[source]["total"] += 1

            # Reject graphs with no entity nodes (can't train on them)
            if "entity" not in graph.node_types or graph["entity"].num_nodes == 0:
                logger.warning("Skipping graph %s — 0 entity nodes", graph_path.name)
                continue

            # Reject graphs with NaN in entity features
            if hasattr(graph["entity"], "x") and torch.isnan(graph["entity"].x).any():
                logger.warning("Skipping graph %s — NaN in entity features", graph_path.name)
                continue

            if self.pre_filter is not None and not self.pre_filter(graph):
                continue

            if self.pre_transform is not None:
                graph = self.pre_transform(graph)

            # ── BUG-FIX #2: Strip non-tensor metadata BEFORE appending ──
            # graph.entity_index (a dict with "UNKNOWN" keys) and
            # graph["patient"].patient_id (a string) crash PyG's
            # InMemoryDataset.collate() which tries to index every attribute.
            graph = _strip_graph_metadata(graph)

            data_list.append(graph)

        if not data_list:
            logger.error("No valid graphs loaded — dataset will be empty!")
            return

        # Collate into a single InMemoryDataset storage
        data, slices = self.collate(data_list)

        # Save processed data to disk so future loads bypass processing
        torch.save((data, slices), self.processed_paths[0])

        # Update internal state
        self.data = data
        self.slices = slices

        # --- per-source summary ---
        for src, stats in source_stats.items():
            if stats["total"] > 0:
                logger.info(
                    "[%s] graphs: %d total — labels FOUND: %d, MISSING: %d",
                    src, stats["total"], stats["found"], stats["missing"],
                )

        # --- class-distribution validation ---
        readmission_vals = []
        for g in data_list:
            if hasattr(g, "y_readmission") and g.y_readmission is not None:
                val = int(g.y_readmission.item()) if g.y_readmission.numel() > 0 else 0
                readmission_vals.append(val)
        if readmission_vals:
            pos = sum(1 for v in readmission_vals if v > 0)
            neg = len(readmission_vals) - pos
            logger.info(
                "Final dataset: %d graphs — class distribution: POS=%d (%.1f%%), NEG=%d (%.1f%%)",
                len(data_list), pos, 100 * pos / len(data_list),
                neg, 100 * neg / len(data_list),
            )
            if pos == 0:
                logger.warning(
                    "All %d graphs have label 0 (no positive readmission cases) — "
                    "training will not learn readmission risk!",
                    len(data_list),
                )
            elif neg == 0:
                logger.warning(
                    "All %d graphs have label 1 (no negative cases) — "
                    "training cannot learn decision boundary!",
                    len(data_list),
                )
        else:
            logger.warning("No y_readmission values found in processed dataset!")

        logger.info(
            "Processed %d graphs into InMemoryDataset (saved to %s) — "
            "labels: %d FOUND, %d MISSING",
            len(data_list), self.processed_paths[0], labels_found, labels_missing,
        )

    def len(self) -> int:
        """Return number of graphs in the dataset.

        Uses the collated InMemoryDataset size when available (after
        processing/filtering), falls back to raw file count otherwise.
        """
        if self._data is not None:
            # slices["patient"]["x"] is a 1-D tensor of cumulative node counts;
            # its length minus 1 equals the number of graphs.
            return self.slices["patient"]["x"].size(0) - 1
        return len(self._graph_files)

    def get(self, idx: int) -> HeteroData:
        """Return a single graph from the collated dataset.

        Uses PyTorch Geometric's native reconstruction logic.
        """

        data = super().get(idx)

        if self.transform is not None:
            data = self.transform(data)

        return data


def collate_fn(batch: list[HeteroData]) -> Batch:
    """Collate function for HeteroData — must use PyG's Batch class.

    HeteroData CANNOT be stacked by default DataLoader — must use
    ``torch_geometric.data.Batch.from_data_list``.

    As a safety measure, each graph in the batch is passed through
    :func:`_strip_graph_metadata` so that any stray non-tensor attributes
    (e.g. ``entity_index``, ``patient_id``) from the get() fallback path
    do not cause PyG's Batch.from_data_list to crash with ``KeyError``.

    Parameters
    ----------
    batch : list[HeteroData]
        List of HeteroData graphs from the dataset.

    Returns
    -------
    Batch
        A PyG Batch object suitable for GNN forward pass.
    """
    # Strip any remaining metadata before batching (defence-in-depth)
    clean_batch = [_strip_graph_metadata(g) for g in batch]
    return Batch.from_data_list(clean_batch)