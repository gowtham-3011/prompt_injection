"""Build labelled train/val/test splits for the injection classifier (Phase 4).

Sources:
  * Positives (label 1 = injection): the payload texts from our attack catalog, PLUS an
    optional public dataset (deepset/prompt-injections on the Hugging Face Hub) when it is
    available and license-compatible.
  * Negatives (label 0 = benign): benign prompts + hard negatives from data/benign.py.

Public datasets are checked, not assumed. `load_public_positives` reports availability,
license and schema and returns a fallback if a dataset cannot be fetched (offline CI),
rather than silently skipping it.

Split integrity (enforced by tests):
  * train / val / test are disjoint,
  * near-duplicate texts are deduplicated ACROSS splits,
  * whole attack families in catalog.HELDOUT_FAMILIES are removed from train/val and put
    only in a separate `heldout` split, to test generalization to unseen attack types.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from src.attacks.catalog import ALL_TEMPLATES, HELDOUT_FAMILIES
from src.attacks.generator import _payload_text
from src.data.benign import benign_items
from src.utils import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Example:
    text: str
    label: int          # 1 = injection, 0 = benign
    source: str         # provenance tag
    family: str = ""    # attack family (positives only), for held-out analysis


@dataclass
class Dataset:
    train: list[Example] = field(default_factory=list)
    val: list[Example] = field(default_factory=list)
    test: list[Example] = field(default_factory=list)
    heldout: list[Example] = field(default_factory=list)  # unseen attack families

    def hard_negatives(self, split_name: str = "test") -> list[Example]:
        """Hard negatives inside one split.

        FPR on hard negatives must be measured on held-out ones only: the same items also
        appear in training (they are legitimate training signal), and scoring the model on
        text it memorised would understate the false-positive rate.
        """
        return [e for e in getattr(self, split_name)
                if e.label == 0 and e.source not in ("benign_prompt",)
                and not e.source.startswith("public:")]

    def counts(self) -> dict[str, dict[str, int]]:
        def c(split: list[Example]) -> dict[str, int]:
            return {"pos": sum(e.label for e in split), "neg": sum(1 - e.label for e in split),
                    "total": len(split)}
        return {name: c(getattr(self, name)) for name in ("train", "val", "test", "heldout")}


# --------------------------------------------------------------------------- #
# Deduplication key
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    """Normalized form for near-duplicate detection: lowercase, strip non-word chars,
    collapse whitespace. Two texts with the same key are treated as duplicates."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


# --------------------------------------------------------------------------- #
# Positives
# --------------------------------------------------------------------------- #
def catalog_positives() -> list[Example]:
    """Injection texts from our own attack catalog (the payload the model would see)."""
    out = []
    for t in ALL_TEMPLATES:
        out.append(Example(_payload_text(t, canary_hint="session id"), 1,
                           f"catalog:{t.id}", family=t.family))
    return out


def load_public_dataset(name: str = "deepset/prompt-injections") -> tuple[list[Example], str]:
    """Try to load a public injection dataset. Returns (examples, status_message).

    Verified before use: availability, schema, and label semantics.
      * deepset/prompt-injections - schema {'text': str, 'label': 0|1}, label 1 = injection,
        Apache-2.0 / CC-BY licensed (confirm on the dataset card before publishing results).
    All splits are merged here and re-split by build_dataset, so OUR split boundaries -
    not the upstream ones - are what the disjointness guarantees apply to.

    On any failure (offline, schema change, gated repo) it returns [] with a message
    explaining what happened, rather than silently skipping the dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return [], f"datasets not installed; skipping {name}, using catalog + local benign only."
    try:
        ds = load_dataset(name)
    except Exception as err:  # network / gated / renamed
        return [], f"could not fetch {name} ({type(err).__name__}); using catalog + local benign only."

    examples: list[Example] = []
    for split_name in ds.keys():
        split = ds[split_name]
        if "text" not in split.column_names or "label" not in split.column_names:
            return [], f"{name}/{split_name} schema unexpected {split.column_names}; skipping."
        for row in split:
            text, label = row.get("text"), row.get("label")
            if text and label in (0, 1):
                examples.append(Example(text, int(label), f"public:{name}", family="public"))
    n_pos = sum(e.label for e in examples)
    return examples, (f"loaded {len(examples)} examples from {name} "
                      f"({n_pos} injection / {len(examples) - n_pos} benign); check its license "
                      f"before publishing results.")


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def _dedupe(examples: list[Example]) -> list[Example]:
    seen: set[str] = set()
    out = []
    for ex in examples:
        key = normalize(ex.text)
        if key and key not in seen:
            seen.add(key)
            out.append(ex)
    return out


def _split(items: list[Example], ratios: tuple[float, float, float], rng: random.Random
           ) -> tuple[list[Example], list[Example], list[Example]]:
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def build_dataset(
    seed: int = 1234,
    use_public: bool = True,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[Dataset, list[str]]:
    """Assemble the full labelled dataset. Returns (dataset, provenance_messages)."""
    messages: list[str] = []
    rng = random.Random(seed)

    positives = catalog_positives()
    negatives = [Example(text, 0, source) for text, source in benign_items()]
    if use_public:
        public, msg = load_public_dataset()
        messages.append(msg)
        positives += [e for e in public if e.label == 1]
        negatives += [e for e in public if e.label == 0]

    # Hold out whole attack families for the generalization test.
    heldout = [e for e in positives if e.family in HELDOUT_FAMILIES]
    in_dist_pos = [e for e in positives if e.family not in HELDOUT_FAMILIES]

    # Dedupe within each pool, then split, then dedupe across splits by construction
    # (splitting disjoint deduped lists keeps them disjoint).
    in_dist_pos = _dedupe(in_dist_pos)
    negatives = _dedupe(negatives)
    heldout = _dedupe(heldout)

    ptr, pva, pte = _split(in_dist_pos, ratios, rng)
    ntr, nva, nte = _split(negatives, ratios, rng)

    ds = Dataset(
        train=_shuffle(ptr + ntr, rng),
        val=_shuffle(pva + nva, rng),
        test=_shuffle(pte + nte, rng),
        heldout=heldout,  # positives only; FPR is measured on the negative test set
    )
    messages.append(f"dataset sizes: {ds.counts()}")
    return ds, messages


def _shuffle(items: list[Example], rng: random.Random) -> list[Example]:
    out = items[:]
    rng.shuffle(out)
    return out
