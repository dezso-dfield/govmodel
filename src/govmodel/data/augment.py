"""Text augmentation for Dutch municipal correspondence.

Citizen letters arrive in every shape imaginable — formal legalese,
NT2/B1, ALL-CAPS rants, rushed typing, dialect, code-switching. The
v0.1 synthetic set covers some of this through Qwen prompts but the
model is still brittle to surface variation that doesn't change the
underlying Awb category.

This module gives you deterministic perturbations so:
  1. The robustness eval pack can quantify *which* surface variants
     the model is fragile to.
  2. The next training cycle can optionally augment the synthetic set
     with label-preserving variants.

Each function is deterministic given a `seed` and label-preserving by
construction. The optional `formal_to_informal` step uses an LLM (lazy
import) — only used for training augmentation, not robustness eval.
"""
from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Character-level typos
# ---------------------------------------------------------------------------


_NL_LETTERS = "abcdefghijklmnopqrstuvwxyzéëïöü"


def inject_typos(text: str, *, rate: float = 0.03, seed: int = 0) -> str:
    """Random char-level edits (swap-with-neighbour / delete / insert)."""
    rng = random.Random(seed)
    chars = list(text)
    out: list[str] = []
    i = 0
    while i < len(chars):
        c = chars[i]
        if c.isalpha() and rng.random() < rate:
            kind = rng.choice(("swap", "delete", "insert"))
            if kind == "swap" and i + 1 < len(chars) and chars[i + 1].isalpha():
                out.append(chars[i + 1])
                out.append(c)
                i += 2
                continue
            if kind == "delete":
                i += 1
                continue
            if kind == "insert":
                out.append(c)
                out.append(rng.choice(_NL_LETTERS))
                i += 1
                continue
        out.append(c)
        i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Casing and punctuation
# ---------------------------------------------------------------------------


def casefold_perturb(text: str, *, kind: str = "lower", seed: int = 0) -> str:
    """kind ∈ {lower, upper, title, random}."""
    if kind == "lower":
        return text.lower()
    if kind == "upper":
        return text.upper()
    if kind == "title":
        return text.title()
    if kind == "random":
        rng = random.Random(seed)
        return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in text)
    raise ValueError(f"unknown kind={kind!r}")


_PUNCT = ",.;:!?"


def punctuation_perturb(text: str, *, mode: str = "strip", seed: int = 0) -> str:
    if mode == "strip":
        return re.sub(rf"[{re.escape(_PUNCT)}]+", "", text)
    if mode == "double":
        return re.sub(rf"([{re.escape(_PUNCT)}])", r"\1\1", text)
    if mode == "random":
        rng = random.Random(seed)
        return "".join("" if c in _PUNCT and rng.random() < 0.5 else c for c in text)
    raise ValueError(f"unknown mode={mode!r}")


# ---------------------------------------------------------------------------
# Shouted emphasis — gov-specific. Angry citizens use ALL-CAPS spans.
# ---------------------------------------------------------------------------


def shout_emphasis(text: str, *, rate: float = 0.10, seed: int = 0) -> str:
    """With probability `rate` per word, uppercase the whole word.

    Letters of complaint frequently contain ALL-CAPS emphasis like
    "ECHT SCHANDALIG" or "MIJN BEZWAAR". The model should treat these
    the same as the lowercase version.
    """
    rng = random.Random(seed)
    return " ".join(w.upper() if rng.random() < rate and w.isalpha() else w
                    for w in text.split(" "))


# ---------------------------------------------------------------------------
# NT2 / B1 simplification — rule-based, deliberately conservative
# ---------------------------------------------------------------------------


_NT2_SUBS: tuple[tuple[str, str], ...] = (
    (r"\becht er\b", "maar"),
    (r"\bechter\b", "maar"),
    (r"\bderhalve\b", "dus"),
    (r"\bvervolgens\b", "daarna"),
    (r"\bdaarentegen\b", "maar"),
    (r"\baangezien\b", "omdat"),
    (r"\bteneinde\b", "om te"),
    (r"\bvolstrekt\b", "helemaal"),
    (r"\bdesalniettemin\b", "toch"),
    (r"\bbijgevolg\b", "dus"),
    (r"\bweliswaar\b", "wel"),
    (r"\bmitsdien\b", "dus"),
    # Formal opener → simple
    (r"\bgeachte heer/mevrouw\b", "beste"),
    (r"\bgeachte heer\b", "beste"),
    (r"\bgeachte mevrouw\b", "beste"),
    (r"\bhoogachtend\b", "groet"),
    (r"\bmet vriendelijke groet\b", "groet"),
)


def nt2_simplify(text: str) -> str:
    """Rewrite formal Dutch connectives to B1/NT2-level surface forms.

    Semantics preserved. Used both for robustness eval (does the
    classifier still see *bezwaar* when the citizen writes simpler
    Dutch?) and for training augmentation.
    """
    out = text
    for pat, rep in _NT2_SUBS:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out


# ---------------------------------------------------------------------------
# Dialect + code-switching — small substitution map
# ---------------------------------------------------------------------------


_DIALECT_SUBS: tuple[tuple[str, str], ...] = (
    # Flemish forms
    (r"\bgij\b", "jij"),
    (r"\bdaar zijn\b", "er zijn"),
    # Common rushed contractions
    (r"\bniet\b", "nie"),
    (r"\bworden\b", "worde"),
    # Anglicisms (citizens sometimes drop English words in)
    (r"\bbecause\b", "omdat"),
    (r"\bbasically\b", "in feite"),
    (r"\bplease\b", "graag"),
    # Turkish / Arabic loan greetings (no semantic change)
    (r"\bselam\b", "hallo"),
    (r"\bsalam\b", "hallo"),
)


def dialect_swap(text: str) -> str:
    """Apply small substitution map for common non-standard variants."""
    out = text
    for pat, rep in _DIALECT_SUBS:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    return out


# ---------------------------------------------------------------------------
# Formal → informal (LLM-based; label-preserving by prompt design)
# ---------------------------------------------------------------------------


def formal_to_informal(text: str, *, client=None) -> str:
    """LLM-driven informal rewrite. Lazy-loads the project's LLMClient.

    Pass `client` to inject a mock for tests; otherwise we instantiate
    `LLMClient.from_env()` and ask it to rewrite the input.
    """
    from govmodel.llm_client import LLMClient  # lazy

    if client is None:
        client = LLMClient.from_env()
    prompt = (
        "Herschrijf onderstaande Nederlandse burgerbrief in informelere, "
        "alledaagsere taal zonder de bedoeling, het type verzoek of de "
        "feitelijke inhoud te veranderen. Antwoord met alleen de herschreven brief.\n\n"
        f"{text}"
    )
    return client.chat(
        [{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=900,
    )


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Augmentation:
    name: str
    fn: Callable[[str], str]
    label_preserving: bool = True


def standard_augmentations(*, seed: int = 0) -> list[Augmentation]:
    """The default robustness-eval pack — label-preserving only.

    `formal_to_informal` is *not* included by default because it depends
    on a running LLM. Add it manually for training augmentation when you
    have an LLM endpoint configured.
    """

    def with_seed(f, k):
        def wrapped(t: str) -> str:
            return f(t, seed=seed + k)
        return wrapped

    def _typos(rate, k):
        return with_seed(lambda t, *, seed: inject_typos(t, rate=rate, seed=seed), k)

    return [
        Augmentation("typos_3pct",     _typos(0.03, 1)),
        Augmentation("typos_8pct",     _typos(0.08, 2)),
        Augmentation("lowercase",      lambda t: casefold_perturb(t, kind="lower")),
        Augmentation("uppercase",      lambda t: casefold_perturb(t, kind="upper")),
        Augmentation("shout_emphasis", with_seed(
            lambda t, *, seed: shout_emphasis(t, rate=0.15, seed=seed), 3)),
        Augmentation("punct_stripped", lambda t: punctuation_perturb(t, mode="strip")),
        Augmentation("punct_doubled",  lambda t: punctuation_perturb(t, mode="double")),
        Augmentation("nt2_simplify",   lambda t: nt2_simplify(t)),
        Augmentation("dialect_swap",   lambda t: dialect_swap(t)),
    ]


def apply_all(text: str, augs: Iterable[Augmentation]) -> list[tuple[str, str]]:
    """Returns `[(perturbation_name, perturbed_text), ...]`."""
    return [(a.name, a.fn(text)) for a in augs]
