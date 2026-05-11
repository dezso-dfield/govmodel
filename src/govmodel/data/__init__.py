"""Dataset hygiene + augmentation utilities for govmodel.

Three pieces:
  - `tools` — validation, dedup, leakage detection, stratified splits.
  - `augment` — deterministic Dutch-flavoured text perturbations.
  - `meta_klacht` — prompt templates for meta-complaint synthetic gen.

Everything here is stdlib + numpy. Torch is only pulled by callers that
actually need embeddings.
"""

from govmodel.data.augment import (
    Augmentation,
    apply_all,
    casefold_perturb,
    dialect_swap,
    formal_to_informal,
    inject_typos,
    nt2_simplify,
    punctuation_perturb,
    shout_emphasis,
    standard_augmentations,
)
from govmodel.data.tools import (
    ValidationIssue,
    balance_report,
    cross_distribution,
    exact_dedupe,
    issues_summary,
    jaccard,
    label_distribution,
    leakage_between,
    multilabel_distribution,
    near_dedupe,
    read_jsonl,
    stratified_split,
    validate_examples,
    write_jsonl,
)

__all__ = [
    "Augmentation",
    "ValidationIssue",
    "apply_all",
    "balance_report",
    "casefold_perturb",
    "cross_distribution",
    "dialect_swap",
    "exact_dedupe",
    "formal_to_informal",
    "inject_typos",
    "issues_summary",
    "jaccard",
    "label_distribution",
    "leakage_between",
    "multilabel_distribution",
    "near_dedupe",
    "nt2_simplify",
    "punctuation_perturb",
    "read_jsonl",
    "shout_emphasis",
    "standard_augmentations",
    "stratified_split",
    "validate_examples",
    "write_jsonl",
]
