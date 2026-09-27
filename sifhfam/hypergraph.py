"""Hyperedge generation (sampled and exhaustive) with deterministic seeds."""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class HyperedgeCollection:
    edges: List[Tuple[int, ...]]
    per_order_sampled: Dict[int, int]
    per_order_total: Dict[int, int]
    K: int
    max_edges: int
    sampled: bool
    seed: int | None

    def counts(self) -> Dict[str, object]:
        return {
            "K": self.K,
            "max_edges": self.max_edges,
            "sampled": self.sampled,
            "n_edges": len(self.edges),
            "per_order_sampled": dict(self.per_order_sampled),
            "per_order_total": dict(self.per_order_total),
            "seed": self.seed,
        }


def theoretical_counts(f: int, K: int) -> Dict[int, int]:
    """Total candidate hyperedge count by order: C(f, k) for k=2..K."""
    out: Dict[int, int] = {}
    for k in range(2, max(2, K) + 1):
        if k <= f:
            out[k] = math.comb(f, k)
        else:
            out[k] = 0
    return out


def generate_hyperedges(
    f: int,
    K: int,
    max_edges: int,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    orders_mode: str = "range_2_K",
) -> HyperedgeCollection:
    """Generate hyperedges of orders 2..K.

    Exhaustive when C(f, k) <= per-order budget for every order; otherwise
    deterministic-budget random sampling per order (the sampler is documented
    and seeded; the realized collection is recorded for reproducibility).

    orders_mode:
      - ``range_2_K``: all orders 2..K (default construction)
      - ``exact_K``: only order K (alternative construction ablation)
    """
    K = int(max(0, K))
    per_order_sampled: Dict[int, int] = {}
    if orders_mode == "exact_K":
        per_order_total = {K: (math.comb(f, K) if K >= 2 and K <= f else 0)} if K >= 2 else {}
    else:
        per_order_total = theoretical_counts(f, K) if K >= 2 else {}
    if K < 2 or f < 2 or max_edges <= 0:
        return HyperedgeCollection([], {}, per_order_total, K, max_edges, False, seed)

    if orders_mode == "exact_K":
        orders = [K] if K <= f else []
    else:
        orders = [k for k in range(2, min(K, f) + 1)]
    if not orders:
        return HyperedgeCollection([], {}, per_order_total, K, max_edges, False, seed)
    per_order_budget = max(1, int(max_edges) // len(orders))
    edges: List[Tuple[int, ...]] = []
    sampled_flag = False

    for order in orders:
        total = per_order_total.get(order, 0)
        if total <= per_order_budget:
            batch = [tuple(c) for c in itertools.combinations(range(f), order)]
            per_order_sampled[order] = len(batch)
            edges.extend(batch)
        else:
            sampled_flag = True
            if rng is None:
                rng = np.random.default_rng(seed)
            seen = set()
            attempts = 0
            max_attempts = per_order_budget * 20
            while len(seen) < per_order_budget and attempts < max_attempts:
                edge = tuple(sorted(int(x) for x in rng.choice(f, size=order, replace=False)))
                seen.add(edge)
                attempts += 1
            batch = sorted(seen)
            per_order_sampled[order] = len(batch)
            edges.extend(batch)

    edges = edges[:max_edges]
    # keep per-order counts consistent with the global cap
    kept: Dict[int, int] = {}
    for e in edges:
        kept[len(e)] = kept.get(len(e), 0) + 1
    return HyperedgeCollection(edges=edges, per_order_sampled=kept,
                               per_order_total=per_order_total, K=K,
                               max_edges=max_edges, sampled=sampled_flag, seed=seed)
