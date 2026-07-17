from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Mention:
    referent_id: str
    mention_idx: int
    update_idx: int
    surface_form: str
    referent_type: str
    start: int
    end: int


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    mentions: tuple[Mention, ...]
    probe_referent: str


def _append(parts: list[str], text: str) -> tuple[int, int]:
    start = sum(len(p) for p in parts)
    parts.append(text)
    return start, start + len(text)


def generate_track_a(*, seed: int = 0, limit_docs: int | None = None) -> list[Document]:
    """Deterministic Track A subset from spec §2.

    This implements the first runnable slice: synthetic state-update interleave
    with gold mention spans.  It deliberately starts small but covers the
    load-bearing axes needed by M1/M2 smoke tests: same-surface vs diff-surface,
    multiple referents, multiple updates, and short/long-ish filler gaps.
    """
    rng = random.Random(seed)
    docs: list[Document] = []
    referent_names = ["Alice", "Boris", "Clara", "Derek", "Elena", "Farid", "Greta", "Hector"]
    places = ["red box", "blue jar", "green bag", "silver cup", "black case", "white drawer"]
    values = ["one", "two", "three", "four", "five", "six"]

    doc_no = 0
    for n_refs in (4, 8, 16):
        for n_updates in (2, 4, 6):
            for gap in ("short", "long"):
                for surface_mode in ("same-surface", "diff-surface"):
                    if limit_docs is not None and len(docs) >= limit_docs:
                        return docs
                    chosen = referent_names[:n_refs]
                    probe = rng.choice(chosen)
                    parts: list[str] = []
                    mentions: list[Mention] = []
                    mention_counts = {name: 0 for name in chosen}
                    filler = " The room is quiet." if gap == "short" else " The room stays quiet while the observer writes a neutral note."
                    for update_idx in range(n_updates):
                        order = chosen[:]
                        rng.shuffle(order)
                        for name in order:
                            value = values[(update_idx + chosen.index(name)) % len(values)]
                            if surface_mode == "same-surface" or update_idx == 0:
                                surface = name
                            else:
                                surface = "the person" if update_idx % 2 else name
                            prefix = "Initially, " if update_idx == 0 else "Later, "
                            _append(parts, prefix)
                            start, end = _append(parts, surface)
                            _append(parts, f" keeps the token in the {places[(update_idx + chosen.index(name)) % len(places)]} and records value {value}.")
                            mentions.append(
                                Mention(
                                    referent_id=name,
                                    mention_idx=mention_counts[name],
                                    update_idx=update_idx,
                                    surface_form=surface,
                                    referent_type="person-location-state",
                                    start=start,
                                    end=end,
                                )
                            )
                            mention_counts[name] += 1
                        _append(parts, filler)
                    _append(parts, f" Question: where is {probe} now?")
                    docs.append(
                        Document(
                            doc_id=f"trackA_{doc_no:04d}_{surface_mode}_{gap}_n{n_refs}_m{n_updates}",
                            text="".join(parts),
                            mentions=tuple(mentions),
                            probe_referent=probe,
                        )
                    )
                    doc_no += 1
    return docs
