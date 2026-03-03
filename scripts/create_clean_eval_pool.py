#!/usr/bin/env python3
"""
Create a clean evaluation pool for GRNformer / GRN evaluation.

This script is a general, framework-agnostic version of the logic used in
the Fair Evaluation Framework in GRNformer_update:

- Takes an expression matrix (to define the gene universe)
- Takes a ground-truth regulatory network (positives)
- Optionally takes one or more files with training negatives
- Builds a clean negative pool as:
    all ordered gene pairs (g1, g2), g1 != g2
    minus all positives
    minus all training negatives (if provided)

Outputs a CSV with columns:
- TF
- Target

This CSV can then be used together with general_grn_evaluation.py
(--clean_eval_dir) or imported into other analysis scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd

from general_grn_evaluation import (
    TFPair,
    load_expression_genes,
    load_positives,
    build_clean_negative_pool,
)


def load_training_negatives_from_files(files: List[str]) -> Set[TFPair]:
    """
    Load training negatives from one or more CSV files.

    Each file is expected to have at least two columns describing pairs, with
    flexible column names:
    - (TF, Target) or (tf, gene) or (Gene1, Gene2)
    or the first two columns are used as a fallback.
    """
    training_neg: Set[TFPair] = set()

    for fp in files:
        df = pd.read_csv(fp)

        # Try common column name variants
        if "TF" in df.columns:
            tf_col = "TF"
        elif "tf" in df.columns:
            tf_col = "tf"
        elif "Gene1" in df.columns:
            tf_col = "Gene1"
        else:
            tf_col = df.columns[0]

        if "Target" in df.columns:
            gene_col = "Target"
        elif "gene" in df.columns:
            gene_col = "gene"
        elif "Gene2" in df.columns:
            gene_col = "Gene2"
        else:
            gene_col = df.columns[1]

        for _, row in df.iterrows():
            tf = str(row[tf_col])
            gene = str(row[gene_col])
            if tf == gene:
                continue
            training_neg.add(TFPair(tf, gene))

    return training_neg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a clean evaluation pool for GRN evaluation.\n\n"
            "The pool is all ordered (gene1, gene2) pairs (gene1 != gene2), "
            "excluding all positives and any training negatives provided."
        )
    )
    parser.add_argument(
        "--expression",
        required=True,
        help="Path to expression matrix CSV (genes x cells, genes in index).",
    )
    parser.add_argument(
        "--network",
        required=True,
        help="Path to ground-truth regulatory network CSV.",
    )
    parser.add_argument(
        "--training_negatives",
        nargs="*",
        default=None,
        help=(
            "Optional one or more CSV files containing training negatives. "
            "Columns should include TF/tf/Gene1 and Target/gene/Gene2, or "
            "the first two columns will be used."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output CSV path for the clean evaluation pool. "
            "Will contain columns 'TF' and 'Target'."
        ),
    )

    args = parser.parse_args()

    # Load gene universe and positives
    genes = load_expression_genes(args.expression)
    genes_universe = set(genes)
    positives = load_positives(args.network, genes_universe)

    # Optional training negatives
    if args.training_negatives:
        training_neg = load_training_negatives_from_files(args.training_negatives)
    else:
        training_neg = None

    # Build clean pool
    clean_pool = build_clean_negative_pool(genes, positives, training_negatives=training_neg)

    # Save to CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([(p.tf, p.gene) for p in clean_pool], columns=["TF", "Target"])
    df.to_csv(out_path, index=False)

    print(f"Clean evaluation pool written to {out_path} (n={len(clean_pool)})")


if __name__ == "__main__":
    main()

