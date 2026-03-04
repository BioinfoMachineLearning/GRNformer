#!/usr/bin/env python3
"""
General GRN evaluation utilities for GRNformer.

Given:
- expression matrix
- regulatory network (ground-truth)
- TF list
- prediction scores (TF, target, score)

This script computes:
- sampled AUROC / AUPRC (using a sampled negative set)
- full-matrix AUROC / AUPRC
- Early Precision (EPR@K) metrics

It is intentionally **framework-agnostic** and supports two usage styles:
- **Explicit clean pool** (recommended): you provide
  - a positives file (e.g. refNetwork or master_test)
  - a clean negatives file (e.g. from create_clean_eval_pool.py)
  - a predictions file
- **Automatic fallback**: you provide
  - a positives file
  - an expression matrix
  and the script builds a clean negative pool internally as all ordered
  gene pairs (g1, g2), g1 != g2, excluding all positives and self-loops.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision


@dataclass(frozen=True)
class TFPair:
    tf: str
    gene: str

    def __str__(self) -> str:
        return f"{self.tf}->{self.gene}"


def _infer_columns(
    df: pd.DataFrame,
    tf_candidates: Sequence[str],
    gene_candidates: Sequence[str],
) -> Tuple[str, str]:
    """Infer TF and gene column names with sensible fallbacks."""
    tf_col = None
    gene_col = None

    for c in tf_candidates:
        if c in df.columns:
            tf_col = c
            break
    for c in gene_candidates:
        if c in df.columns:
            gene_col = c
            break

    if tf_col is None or gene_col is None:
        if len(df.columns) < 2:
            raise ValueError("Expected at least 2 columns for TF/Target.")
        tf_col = df.columns[0]
        gene_col = df.columns[1]

    return tf_col, gene_col


def load_expression_genes(expr_path: str) -> List[str]:
    """Load gene names from an expression matrix (genes x cells)."""
    df = pd.read_csv(expr_path, index_col=0)
    # Genes are assumed to be in the index
    genes = df.index.astype(str).tolist()
    return genes


def load_tfs(tf_path: str, genes_universe: Set[str]) -> List[str]:
    """
    Load TF list and intersect with the expression gene universe.

    Accepts files with either a header containing 'TF' or a single column CSV.
    """
    df = pd.read_csv(tf_path)
    if "TF" in df.columns:
        tf_col = "TF"
    else:
        # Fall back to first column
        tf_col = df.columns[0]

    all_tfs = set(df[tf_col].astype(str).tolist())
    # Filter to TFs present in expression genes
    tfs = sorted(all_tfs.intersection(genes_universe))
    return tfs


def load_positives(
    network_path: str,
    genes_universe: Set[str],
) -> Set[TFPair]:
    """
    Load ground-truth regulatory network as a set of TFPair.

    The network file can have columns:
    - ('TF', 'Target')  (preferred)
    - ('tf', 'gene')
    - or any first two columns (fallback).
    Pairs are filtered so that both TF and target are in the gene universe
    and self-loops are removed.
    """
    df = pd.read_csv(network_path)

    tf_col, gene_col = _infer_columns(
        df, tf_candidates=("TF", "tf", "Gene1"), gene_candidates=("Target", "gene", "Gene2")
    )

    positives: Set[TFPair] = set()
    for _, row in df.iterrows():
        tf = str(row[tf_col])
        gene = str(row[gene_col])
        if tf == gene:
            continue
        if tf in genes_universe and gene in genes_universe:
            positives.add(TFPair(tf, gene))

    return positives


def load_pairs_from_file(
    path: str,
    genes_universe: Optional[Set[str]] = None,
    positives_only_if_labeled: bool = False,
) -> Set[TFPair]:
    """
    Load TF-gene pairs from a generic CSV file.

    - Infers TF/Target columns via common name patterns or first two columns.
    - If a label column ('label' or 'Label') exists and positives_only_if_labeled
      is True, only rows with label == 1 are used.
    - Self-loops are removed.
    - If genes_universe is provided, both TF and gene must be in that set.
    """
    df = pd.read_csv(path)

    tf_col, gene_col = _infer_columns(
        df, tf_candidates=("TF", "tf", "Gene1"), gene_candidates=("Target", "gene", "Gene2", "Gene")
    )

    if positives_only_if_labeled and ("label" in df.columns or "Label" in df.columns):
        label_col = "label" if "label" in df.columns else "Label"
        df = df[df[label_col] == 1]

    pairs: Set[TFPair] = set()
    for _, row in df.iterrows():
        tf = str(row[tf_col])
        gene = str(row[gene_col])
        if tf == gene:
            continue
        if genes_universe is not None and (tf not in genes_universe or gene not in genes_universe):
            continue
        pairs.add(TFPair(tf, gene))

    return pairs


def load_predictions(
    preds_path: str,
    genes_universe: Set[str],
) -> Dict[TFPair, float]:
    """
    Load predictions as a mapping TFPair -> score.

    The prediction file is expected to have at least three columns:
    - TF / tf / Gene1
    - Target / gene / Gene2
    - score / Score / weight / Weight / prob / Prob
    or, as a fallback, the first three columns will be interpreted as
    (tf, gene, score).
    """
    df = pd.read_csv(preds_path)

    tf_col, gene_col = _infer_columns(
        df, tf_candidates=("TF", "tf", "Gene1"), gene_candidates=("Target", "gene", "Gene", "Gene2")
    )

    score_col = None
    # Common score/probability column names across pipelines
    # (GRNformer writes 'probability')
    for c in (
        "probability",
        "Probability",
        "score",
        "Score",
        "weight",
        "Weight",
        "prob",
        "Prob",
        "prediction",
        "Prediction",
    ):
        if c in df.columns:
            score_col = c
            break

    # Fallbacks
    if score_col is None:
        # Use the last column as score if not found explicitly.
        # Note: if the last column is a rank (smaller is better), you should
        # rename/select the probability column instead.
        if len(df.columns) < 3:
            raise ValueError(
                f"Predictions file {preds_path} must have at least 3 columns."
            )
        score_col = df.columns[-1]

    preds: Dict[TFPair, float] = {}
    for _, row in df.iterrows():
        tf = str(row[tf_col])
        gene = str(row[gene_col])
        if tf == gene:
            continue
        if tf not in genes_universe or gene not in genes_universe:
            continue
        try:
            score = float(row[score_col])
        except Exception:
            # Skip rows where score cannot be parsed
            continue
        preds[TFPair(tf, gene)] = score

    return preds


def build_full_negative_universe(
    genes_universe: Sequence[str],
    positives: Set[TFPair],
) -> Set[TFPair]:
    """
    Build full negative universe as all ordered gene pairs (g1, g2) with g1 != g2,
    excluding known positives.
    """
    gene_set = list(dict.fromkeys(str(g) for g in genes_universe))
    positives_set = set(positives)

    negatives: Set[TFPair] = set()
    for g1 in gene_set:
        for g2 in gene_set:
            if g1 == g2:
                continue
            pair = TFPair(g1, g2)
            if pair not in positives_set:
                negatives.add(pair)
    return negatives


def build_clean_negative_pool(
    genes_universe: Sequence[str],
    positives: Set[TFPair],
    training_negatives: Optional[Set[TFPair]] = None,
) -> Set[TFPair]:
    """
    Build a clean negative pool:
    - all ordered gene pairs (g1, g2), g1 != g2
    - excluding all known positives
    - excluding any training negatives if provided

   
    """
    gene_set = list(dict.fromkeys(str(g) for g in genes_universe))
    positives_set = set(positives)
    training_neg_set: Set[TFPair] = training_negatives or set()

    negatives: Set[TFPair] = set()
    for g1 in gene_set:
        for g2 in gene_set:
            if g1 == g2:
                continue
            pair = TFPair(g1, g2)
            if pair in positives_set:
                continue
            if pair in training_neg_set:
                continue
            negatives.add(pair)
    return negatives


def sample_negatives(
    negatives: Sequence[TFPair],
    n_samples: int,
    seed: Optional[int] = None,
) -> Set[TFPair]:
    """Sample a subset of negatives without replacement."""
    if seed is not None:
        np.random.seed(seed)

    negatives = list(negatives)
    if n_samples >= len(negatives):
        return set(negatives)

    idx = np.random.choice(len(negatives), size=n_samples, replace=False)
    return {negatives[i] for i in idx}


def compute_auroc_aupr(
    positives: Set[TFPair],
    negatives: Set[TFPair],
    predictions: Dict[TFPair, float],
) -> Tuple[Optional[float], Optional[float], int, int]:
    """
    Compute AUROC and AUPRC for given positive and negative sets.

    Only pairs that have predictions are used.
    """
    pos_scores: List[float] = []
    neg_scores: List[float] = []

    for p in positives:
        if p in predictions:
            pos_scores.append(predictions[p])
    for n in negatives:
        if n in predictions:
            neg_scores.append(predictions[n])

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        return None, None, n_pos, n_neg

    # Build label and score tensors
    y_true = torch.cat(
        [
            torch.ones(n_pos, dtype=torch.int64),
            torch.zeros(n_neg, dtype=torch.int64),
        ],
        dim=0,
    )
    y_scores = torch.tensor(pos_scores + neg_scores, dtype=torch.float32)

    # Torchmetrics AUROC and AUPR (average precision)
    auroc_metric = BinaryAUROC()
    aupr_metric = BinaryAveragePrecision()

    auroc = auroc_metric(y_scores, y_true).item()
    aupr = aupr_metric(y_scores, y_true).item()
    return auroc, aupr, n_pos, n_neg


def calculate_epr(
    predictions: Dict[TFPair, float],
    test_positives: Set[TFPair],
    test_negatives: Set[TFPair],
    k_values: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    """
    Calculate Early Precision (EPR@K) metrics for top K ground-truth positives.

    This is adapted from the EPR utilities in GRNformer_update but made
    framework-agnostic.
    """
    gt_k = len(test_positives)
    if gt_k == 0:
        return {}

    if k_values is None:
        k_values = [gt_k]
    else:
        k_values = list(k_values)
        if gt_k not in k_values:
            k_values.append(gt_k)

    pos_scores: List[float] = []
    neg_scores: List[float] = []
    pos_pairs: List[TFPair] = []
    neg_pairs: List[TFPair] = []

    for pair in test_positives:
        if pair in predictions:
            pos_scores.append(predictions[pair])
            pos_pairs.append(pair)

    for pair in test_negatives:
        if pair in predictions:
            neg_scores.append(predictions[pair])
            neg_pairs.append(pair)

    if len(pos_pairs) == 0:
        return {}

    all_pairs = pos_pairs + neg_pairs
    all_scores = pos_scores + neg_scores

    # Sort by score descending
    sorted_idx = np.argsort(all_scores)[::-1]
    sorted_pairs = [all_pairs[i] for i in sorted_idx]

    positive_set = set(test_positives)

    epr_metrics: Dict[str, float] = {}
    for k in k_values:
        if k <= 0:
            continue
        k_actual = min(k, len(sorted_pairs))
        if k_actual == 0:
            continue
        top_k = sorted_pairs[:k_actual]
        tp_in_top_k = sum(1 for pair in top_k if pair in positive_set)
        epr = tp_in_top_k / float(k_actual)
        epr_metrics[f"epr@{k}"] = epr

    return epr_metrics


def evaluate_grn(
    positives_path: str,
    clean_negatives_path: Optional[str],
    preds_path: str,
    expr_path: Optional[str] = None,
    sampled_neg_ratio: float = 1.0,
    sampled_seed: Optional[int] = 42,
    epr_k: Optional[Sequence[int]] = None,
    tf_path: Optional[str] = None,
    covered_edges_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    High-level evaluation entry point.

    Returns a dict with:
    - sampled_auroc, sampled_aupr
    - full_auroc, full_aupr
    - counts for positives/negatives used
    - EPR@K entries (e.g. 'epr@100')

    The caller is responsible for providing:
    - `positives_path`: ground-truth edges (all positives, or a labeled file
      like master_test; if a label column exists, only label==1 rows are used).
    - Either:
        * `clean_negatives_path`: precomputed clean negative pool (e.g. from
          create_clean_eval_pool.py), OR
        * `expr_path`: expression matrix, in which case a clean pool is built
          automatically as all ordered gene pairs minus positives and
          self-loops.
    - `preds_path`: prediction scores for TF-target pairs.

    If `expr_path` is given, genes are restricted to the expression universe.
    """
    # Optional expression universe
    genes_universe: Optional[Set[str]] = None
    genes: Optional[List[str]] = None
    if expr_path is not None:
        genes = load_expression_genes(expr_path)
        genes_universe = set(genes)

    # Optional TF universe (for restricting positives to TF->gene)
    tfs_universe: Optional[Set[str]] = None
    if tf_path is not None:
        if genes_universe is not None:
            # Intersect TFs with expression genes
            tfs_universe = set(load_tfs(tf_path, genes_universe))
        else:
            tf_df = pd.read_csv(tf_path)
            if "TF" in tf_df.columns:
                tf_col = "TF"
            else:
                tf_col = tf_df.columns[0]
            tfs_universe = set(tf_df[tf_col].astype(str).tolist())

    # Load positives (optionally filtered by expression genes)
    positives = load_pairs_from_file(
        positives_path,
        genes_universe=genes_universe,
        positives_only_if_labeled=True,
    )

    # Restrict positives to TF->gene if we have a TF universe
    if tfs_universe is not None:
        positives = {p for p in positives if p.tf in tfs_universe}

    # Total positives in ground-truth file after TF/expression filtering
    total_positives_in_file = len(positives)

    # Optionally restrict positives to edges actually covered by TFwalker input
    if covered_edges_path is not None:
        covered_df = pd.read_csv(covered_edges_path)
        # Infer TF / gene columns for covered edges
        if {"Gene1", "Gene2"}.issubset(covered_df.columns):
            cov_tf_col, cov_gene_col = "Gene1", "Gene2"
        else:
            cov_tf_col, cov_gene_col = covered_df.columns[0], covered_df.columns[1]

        covered_pairs = {
            TFPair(str(row[cov_tf_col]), str(row[cov_gene_col]))
            for _, row in covered_df.iterrows()
        }

        positives = positives.intersection(covered_pairs)

    # Determine negatives
    if clean_negatives_path is not None:
        # Explicit clean pool provided
        full_negatives = load_pairs_from_file(
            clean_negatives_path,
            genes_universe=genes_universe,
            positives_only_if_labeled=False,
        )

        # If no expression universe was provided, derive gene universe from pairs
        if genes_universe is None:
            genes_universe = set()
            for p in positives:
                genes_universe.add(p.tf)
                genes_universe.add(p.gene)
            for n in full_negatives:
                genes_universe.add(n.tf)
                genes_universe.add(n.gene)
    else:
        # No clean negatives file: build clean pool internally
        if genes is None:
            raise ValueError(
                "When --clean_negatives is not provided, --expression must be set "
                "to define the gene universe for automatic clean pool construction."
            )
        full_negatives = build_clean_negative_pool(genes, positives)
        # genes_universe is already set from expression

    # Load predictions filtered to the gene universe
    preds = load_predictions(preds_path, genes_universe)

    # Sampled evaluation: sample negatives at given ratio relative to #positives
    n_pos = len(positives)
    n_sampled_neg = int(round(sampled_neg_ratio * n_pos))

    # Single-sample metrics (for backward compatibility)
    sampled_negatives_single = sample_negatives(
        list(full_negatives), n_samples=n_sampled_neg, seed=sampled_seed
    )
    sampled_auroc_single, sampled_aupr_single, n_pos_s, _ = compute_auroc_aupr(
        positives, sampled_negatives_single, preds
    )

    # Coverage: fraction of ground-truth positives that are actually evaluated
    coverage = (
        n_pos_s / float(total_positives_in_file)
        if total_positives_in_file > 0
        else 0.0
    )

    # Bootstrapped sampled metrics (sampling negatives from clean pool without replacement per bootstrap draw)
    B = 100
    neg_list = list(full_negatives)
    auroc_vals: List[float] = []
    aupr_vals: List[float] = []

    if len(neg_list) > 0 and n_sampled_neg > 0:
        rng = np.random.default_rng(sampled_seed)
        for _ in range(B):
            if n_sampled_neg >= len(neg_list):
                # If requested sample size exceeds pool, fall back to full pool
                boot_negatives = set(neg_list)
            else:
                # Sample each bootstrap set without replacement
                idx = rng.choice(len(neg_list), size=n_sampled_neg, replace=False)
                boot_negatives = {neg_list[i] for i in idx}
            auroc_b, aupr_b, _, _ = compute_auroc_aupr(positives, boot_negatives, preds)
            if auroc_b is not None and aupr_b is not None:
                auroc_vals.append(auroc_b)
                aupr_vals.append(aupr_b)

    if auroc_vals and aupr_vals:
        sampled_auroc_mean = float(np.mean(auroc_vals))
        sampled_auroc_std = float(np.std(auroc_vals, ddof=1))
        sampled_aupr_mean = float(np.mean(aupr_vals))
        sampled_aupr_std = float(np.std(aupr_vals, ddof=1))
    else:
        sampled_auroc_mean = sampled_auroc_std = None
        sampled_aupr_mean = sampled_aupr_std = None

    full_auroc, full_aupr, n_pos_f, n_neg_f = compute_auroc_aupr(
        positives, full_negatives, preds
    )

    # EPR: by default at K = number of positives
    epr_metrics = calculate_epr(
        predictions=preds,
        test_positives=positives,
        test_negatives=full_negatives,
        k_values=epr_k,
    )

    results: Dict[str, float] = {
        "n_positives": float(n_pos_f),
        "n_full_negatives": float(n_neg_f),
        "n_sampled_negatives": float(n_sampled_neg),
        "bootstrap_repeats": float(B),
        "total_positives_in_file": float(total_positives_in_file),
        "n_positives_with_predictions": float(n_pos_s),
        "positive_coverage": float(coverage),
    }

    # Single-sample metrics
    if sampled_auroc_single is not None:
        results["sampled_auroc"] = float(sampled_auroc_single)
    if sampled_aupr_single is not None:
        results["sampled_aupr"] = float(sampled_aupr_single)

    # Bootstrapped metrics
    if sampled_auroc_mean is not None:
        results["sampled_auroc_mean"] = sampled_auroc_mean
        results["sampled_auroc_std"] = sampled_auroc_std
    if sampled_aupr_mean is not None:
        results["sampled_aupr_mean"] = sampled_aupr_mean
        results["sampled_aupr_std"] = sampled_aupr_std
    if full_auroc is not None:
        results["full_auroc"] = float(full_auroc)
    if full_aupr is not None:
        results["full_aupr"] = float(full_aupr)

    # Merge EPR metrics
    for k, v in epr_metrics.items():
        results[k] = float(v)

    return results


def parse_k_values(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or raw == "":
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [int(p) for p in parts]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "General GRN evaluation for GRNformer.\n\n"
            "Given positives, clean negatives, and prediction scores, "
            "computes sampled/full AUROC/AUPR and EPR metrics.\n\n"
            "Positives can be a refNetwork-style file or a master_test-style "
            "file (if a label column is present, only label==1 rows are used). "
            "Clean negatives should be a precomputed clean pool, e.g. from "
            "scripts/create_clean_eval_pool.py."
        )
    )
    parser.add_argument(
        "--expression",
        required=False,
        help="Path to expression matrix CSV (genes x cells, genes in index).",
    )
    parser.add_argument(
        "--tfs",
        required=False,
        help=(
            "Path to TF list CSV (same file used for GRNFormer, e.g. TFs.csv). "
            "If provided, positives will be restricted to TF->gene edges "
            "(TFs intersected with expression genes when --expression is given)."
        ),
    )
    parser.add_argument(
        "--covered_edges",
        required=False,
        help=(
            "Optional CSV listing TF->gene pairs covered by TFwalker input "
            "(e.g. edge_index_unique mapped to Gene1,Gene2). If provided, "
            "positives are restricted to the intersection of this set and the "
            "ground-truth positives, and coverage is reported."
        ),
    )
    parser.add_argument(
        "--positives",
        required=True,
        help=(
            "Path to ground-truth positives CSV (e.g. refNetwork.csv or "
            "master_test.csv). If the file has a label column, only label==1 "
            "rows are treated as positives."
        ),
    )
    parser.add_argument(
        "--clean_negatives",
        required=False,
        help=(
            "Path to clean negative pool CSV (e.g. output of "
            "scripts/create_clean_eval_pool.py). If omitted, a clean pool "
            "is built automatically from --expression and --positives by "
            "taking all ordered gene pairs minus positives and self-loops."
        ),
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions CSV (TF, Target, score).",
    )
    parser.add_argument(
        "--sampled_neg_ratio",
        type=float,
        default=1.0,
        help="Ratio of sampled negatives to positives for sampled evaluation.",
    )
    parser.add_argument(
        "--sampled_seed",
        type=int,
        default=42,
        help="Random seed for negative sampling.",
    )
    parser.add_argument(
        "--epr_k",
        type=str,
        default=None,
        help=(
            "Comma-separated list of K values for EPR@K. "
            "If omitted, uses K = number of positives."
        ),
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save metrics as JSON.",
    )

    args = parser.parse_args()

    epr_k = parse_k_values(args.epr_k)
    results = evaluate_grn(
        positives_path=args.positives,
        clean_negatives_path=args.clean_negatives,
        preds_path=args.predictions,
        expr_path=args.expression,
        sampled_neg_ratio=args.sampled_neg_ratio,
        sampled_seed=args.sampled_seed,
        epr_k=epr_k,
        tf_path=args.tfs,
        covered_edges_path=args.covered_edges,
    )

    # Print results to stdout
    print(json.dumps(results, indent=2, sort_keys=True))

    # Optionally save to JSON file
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()

