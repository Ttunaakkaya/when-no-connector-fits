"""Stage 5: matched candidate blocks (master plan section 5).

For each eligible peg, four blocks share the peg, its renders and the candidate E2, so that only
difficulty and mate presence change:

    present / easy  [M,  E1, E2]
    present / hard  [M,  H,  E2]
    absent  / easy  [E3, E1, E2]
    absent  / hard  [E3, H,  E2]

Declared selection rules, fixed before any block is built or scored:

- Similarity is geometric only, never a model score: sim(A, B) = max over the four permitted quarter
  turns of area(A and R(B)) / area(A or R(B)), with both openings on their own centroid and at true
  millimetre scale. No normalisation, so a size difference lowers similarity.
- M is the peg's own (nominal) socket.
- H is the *most similar verified incompatible* opening, chosen between (a) catalogue sockets in the
  peg's split partition and (b) a constructed look-alike: the mate's opening shrunk to the smallest
  interference in SHRINK_LADDER whose label is incompatible with margin <= -2 * EPSILON. Whichever
  has the higher similarity to the mate wins, so a genuinely close catalogue socket is not passed
  over, and a peg is never rejected merely because its partition holds no close neighbour.
- E1, E2, E3 are the three least similar verified incompatible catalogue sockets in the partition,
  required only to differ from each other (pairwise sim <= EASY_MAX_PAIR_SIM), ordered by id. The
  rule is relative rather than an absolute similarity cap, because what the design needs is that H
  is closer to the mate than the easy candidates are, and the achievable similarity depends on the
  partition: the test families are mostly round, so even a circle against an octagon overlaps
  heavily. (An absolute cap of 0.5 was tried first and left only 4 of 30 test pegs eligible; the
  rule was revised on geometric yield alone, before any scoring.)
- A peg is eligible only if its difficulty tiers actually separate: sim(mate, H) >= MIN_HARD_SIM and
  sim(mate, H) - max_i sim(mate, E_i) >= MIN_DIFFICULTY_GAP. Otherwise it is logged as ineligible.
- Candidates come only from the peg's own partition. Ambiguous pairs and loose-compatible sockets
  are never eligible, and their stored labels are unchanged.
- Every block's candidate labels are re-verified after construction: a present block must hold
  exactly one compatible candidate, an absent block none.
- Presentation order is shuffled per block from CANDIDATE_SEED and recorded.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from shapely import Polygon
from shapely.affinity import rotate

from wncf.geometry import EPSILON, QUARTER_TURNS, label_pair
from wncf.parts import Part, nominal_bore

CANDIDATE_SEED = 20260923
SHRINK_LADDER = (0.2, 0.3, 0.5, 0.8)  # mm of interference per side for a constructed look-alike
HARD_MAX_MARGIN = -2 * EPSILON  # a constructed look-alike must be this clearly incompatible
EASY_MAX_PAIR_SIM = 0.8  # easy distractors must differ from each other
MIN_HARD_SIM = 0.6  # H must really resemble the mate
MIN_DIFFICULTY_GAP = 0.15  # H must be closer to the mate than any easy distractor, by this much
REGIMES = ("present_easy", "present_hard", "absent_easy", "absent_hard")


@dataclass(frozen=True)
class Candidate:
    cand_id: str
    bore: Polygon
    kind: str  # catalog | shrunk
    role: str  # M | H | E1 | E2 | E3
    label: str  # verified against this block's peg
    margin_mm: float
    sim_to_mate: float


@dataclass(frozen=True)
class Block:
    block_id: str
    peg_id: str
    family: str
    split: str
    regime: str
    candidates: tuple[Candidate, ...]  # presentation order
    mate_index: int | None


def similarity(a: Polygon, b: Polygon) -> float:
    """Largest overlap between two openings over the permitted quarter turns, at true scale."""
    best = 0.0
    for deg in QUARTER_TURNS:
        rb = rotate(b, deg, origin=(0, 0))
        union = a.union(rb).area
        if union > 0:
            best = max(best, a.intersection(rb).area / union)
    return best


def shrunk_look_alike(part: Part) -> tuple[Candidate, None] | tuple[None, str]:
    """The mate's own opening, shrunk to the smallest interference that is clearly incompatible."""
    for t in SHRINK_LADDER:
        bore = nominal_bore(part.peg, -t)
        lab = label_pair(part.peg, bore)
        if lab.label == "incompatible" and lab.margin_mm <= HARD_MAX_MARGIN:
            return Candidate(f"{part.part_id}_shrunk{t:.2f}", bore, "shrunk", "H", lab.label,
                             lab.margin_mm, similarity(part.bore, bore)), None
    return None, "no shrink level in the ladder is clearly incompatible"


def build_blocks(parts: list[Part], splits: dict[str, str], labels: dict[tuple[str, str], tuple[str, float]],
                 ) -> tuple[list[Block], list[dict]]:
    """Returns the blocks and one log row per peg (eligible or not, with the reason)."""
    by_split: dict[str, list[Part]] = {}
    for p in parts:
        by_split.setdefault(splits[p.family], []).append(p)

    blocks, log = [], []
    for part in parts:
        split = splits[part.family]
        row = {"peg_id": part.part_id, "family": part.family, "split": split, "eligible": 0, "reason": "",
               "h_kind": "", "h_sim": "", "h_margin_mm": "", "easy_max_sim": "", "n_incompatible_pool": 0}

        mate_label, mate_margin = labels[part.part_id, part.part_id]
        if mate_label != "compatible":
            row["reason"] = f"own socket is {mate_label}"
            log.append(row)
            continue

        pool = []
        for q in by_split[split]:
            if q.part_id == part.part_id:
                continue
            lab, margin = labels[part.part_id, q.part_id]
            if lab != "incompatible":  # ambiguous and loose-compatible sockets are never eligible
                continue
            pool.append((q, margin, similarity(part.bore, q.bore)))
        row["n_incompatible_pool"] = len(pool)

        easy = _pick_easy(pool)
        if easy is None:
            row["reason"] = f"fewer than 3 mutually distinct incompatible sockets in {split} (pool {len(pool)})"
            log.append(row)
            continue

        hard, why = _pick_hard(part, pool)
        if hard is None:
            row["reason"] = why
            log.append(row)
            continue

        easy_max = max(e.sim_to_mate for e in easy)
        row.update(h_kind=hard.kind, h_sim=round(hard.sim_to_mate, 4),
                   h_margin_mm=round(hard.margin_mm, 4), easy_max_sim=round(easy_max, 4))
        if hard.sim_to_mate < MIN_HARD_SIM:
            row["reason"] = f"hard candidate resembles the mate too little (sim {hard.sim_to_mate:.2f})"
            log.append(row)
            continue
        if hard.sim_to_mate - easy_max < MIN_DIFFICULTY_GAP:
            row["reason"] = (f"difficulty tiers do not separate (hard {hard.sim_to_mate:.2f} vs "
                             f"easiest-worst {easy_max:.2f})")
            log.append(row)
            continue

        mate = Candidate(part.part_id, part.bore, "catalog", "M", mate_label, mate_margin, 1.0)
        e1, e2, e3 = easy
        row["eligible"] = 1
        log.append(row)

        for regime in REGIMES:
            first = mate if regime.startswith("present") else e3
            second = hard if regime.endswith("hard") else e1
            cands = _ordered([first, second, e2], f"{part.part_id}_{regime}")
            mate_idx = next((i for i, c in enumerate(cands) if c.label == "compatible"), None)
            blocks.append(Block(f"{part.part_id}_{regime}", part.part_id, part.family, split, regime,
                                tuple(cands), mate_idx))
    return blocks, log


def _pick_easy(pool) -> tuple[Candidate, Candidate, Candidate] | None:
    chosen: list[tuple] = []
    for q, margin, sim in sorted(pool, key=lambda t: (t[2], t[0].part_id)):  # least similar first
        if any(similarity(q.bore, other.bore) > EASY_MAX_PAIR_SIM for other, _, _ in chosen):
            continue
        chosen.append((q, margin, sim))
        if len(chosen) == 3:
            break
    if len(chosen) < 3:
        return None
    chosen.sort(key=lambda t: t[0].part_id)
    return tuple(Candidate(q.part_id, q.bore, "catalog", f"E{i}", "incompatible", margin, sim)
                 for i, (q, margin, sim) in enumerate(chosen, start=1))


def _pick_hard(part: Part, pool) -> tuple[Candidate, None] | tuple[None, str]:
    constructed, why = shrunk_look_alike(part)
    best_catalog = max(pool, key=lambda t: (t[2], t[0].part_id), default=None)
    catalog = None
    if best_catalog is not None:
        q, margin, sim = best_catalog
        catalog = Candidate(q.part_id, q.bore, "catalog", "H", "incompatible", margin, sim)
    options = [c for c in (constructed, catalog) if c is not None]
    if not options:
        return None, why or "no incompatible candidate available"
    return max(options, key=lambda c: c.sim_to_mate), None


def _ordered(cands: list[Candidate], block_id: str) -> list[Candidate]:
    order = list(cands)
    random.Random(f"{CANDIDATE_SEED}:{block_id}").shuffle(order)
    return order


def verify(block: Block) -> None:
    """Re-verify a finished block against the oracle contract."""
    n_compatible = sum(c.label == "compatible" for c in block.candidates)
    expected = 1 if block.regime.startswith("present") else 0
    if n_compatible != expected:
        raise AssertionError(f"{block.block_id}: {n_compatible} compatible candidates, expected {expected}")
    if len({c.cand_id for c in block.candidates}) != 3:
        raise AssertionError(f"{block.block_id}: candidates are not distinct")
