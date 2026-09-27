"""Submodular coverage objective and greedy selection (prompt sections 4 and 6).

Scope statement enforced by design
----------------------------------
This is a *representative hyperedge coverage* objective (plus a modular vertex
term).  It is NOT a synergy-recovery objective.  The binary cover g(k)=1[k>=1]
rewards covering a jointly-scored group once; concave soft/fractional covers
give additional (diminishing) marginal gain to later group members.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np


# --------------------------- coverage functions --------------------------- #

def g_binary(k: int) -> float:
    """g(k) = 1[k >= 1]  (legacy/reference representative coverage)."""
    return 1.0 if k >= 1 else 0.0


def g_soft(k: int, rho: float = 0.5) -> float:
    """g(k) = 1 - (1-rho)^k : monotone, discrete-concave, rho in (0,1]."""
    if rho <= 0:
        return g_binary(k)
    return 1.0 - (1.0 - float(rho)) ** k


def g_fraction(k: int, edge_size: int) -> float:
    """g(k, |e|) = min(k / |e|, 1) : monotone, discrete-concave."""
    if edge_size <= 0:
        return 0.0
    return float(min(k / float(edge_size), 1.0))


def make_g(g_type: str, soft_rho: float = 0.5):
    if g_type == "binary":
        return lambda k, edge_size=2: g_binary(k)
    if g_type == "soft":
        return lambda k, edge_size=2: g_soft(k, soft_rho)
    if g_type == "fraction":
        return lambda k, edge_size=2: g_fraction(k, edge_size)
    raise ValueError(f"Unknown g_type: {g_type}")


# ------------------------------ objective ------------------------------ #

@dataclass
class Objective:
    """F(S) = alpha * sum_{j in S} mu_V[j] + beta * sum_e w_e * g(|S ∩ e|, |e|)."""

    vertex_mu: np.ndarray
    edge_weights: Dict[Tuple[int, ...], float]
    alpha: float = 1.0
    beta: float = 1.0
    g_type: str = "binary"
    soft_rho: float = 0.5
    edges_by_vertex: Dict[int, List[Tuple[int, ...]]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.edges_by_vertex:
            f = len(self.vertex_mu)
            inc: Dict[int, List[Tuple[int, ...]]] = {i: [] for i in range(f)}
            for e, w in self.edge_weights.items():
                if w <= 0:
                    continue
                for j in e:
                    if j < f:
                        inc[j].append(e)
            self.edges_by_vertex = inc
        self._g = make_g(self.g_type, self.soft_rho)

    def value(self, S: Sequence[int]) -> float:
        S_set = set(int(x) for x in S)
        val = self.alpha * float(sum(self.vertex_mu[j] for j in S_set))
        for e, w in self.edge_weights.items():
            if w <= 0:
                continue
            k = sum(1 for j in e if j in S_set)
            if k > 0:
                val += self.beta * w * self._g(k, len(e))
        return float(val)

    def marginal(self, i: int, k_e: Dict[Tuple[int, ...], int]) -> float:
        """Marginal gain of adding i given current coverage counts k_e."""
        g = self.alpha * float(self.vertex_mu[i])
        for e in self.edges_by_vertex.get(i, ()):
            w = self.edge_weights[e]
            if w <= 0:
                continue
            k_old = k_e.get(e, 0)
            g += self.beta * w * (self._g(k_old + 1, len(e)) - self._g(k_old, len(e)))
        return g


# ------------------------------- greedy -------------------------------- #

@dataclass
class GreedyResult:
    order: List[int]           # full greedy path (ranking)
    prefix_values: List[float] # F of each prefix
    marginal_gains: List[float]
    prefix_efficiency: List[float]


def greedy_path(obj: Objective, budget: int, stop_nonpositive: bool = True) -> GreedyResult:
    """Standard marginal-gain greedy; returns the full path as a ranking."""
    f = len(obj.vertex_mu)
    k_e: Dict[Tuple[int, ...], int] = {e: 0 for e, w in obj.edge_weights.items() if w > 0}
    remaining = set(range(f))
    order: List[int] = []
    gains: List[float] = []
    vals: List[float] = []
    acc = 0.0
    for _ in range(int(min(budget, f))):
        if not remaining:
            break
        best_i, best_g = None, -np.inf
        for i in remaining:
            g = obj.marginal(i, k_e)
            if g > best_g:
                best_g, best_i = g, i
        if best_i is None or (stop_nonpositive and best_g <= 0):
            break
        order.append(int(best_i))
        remaining.remove(best_i)
        gains.append(float(best_g))
        acc += float(best_g)
        vals.append(acc)
        for e in obj.edges_by_vertex.get(best_i, ()):
            if e in k_e:
                k_e[e] += 1
    eff = [vals[t] / (t + 1) for t in range(len(vals))]
    return GreedyResult(order=order, prefix_values=vals, marginal_gains=gains,
                        prefix_efficiency=eff)


def select_by_cardinality(path: GreedyResult, m: int) -> List[int]:
    return path.order[: int(m)]


def select_by_efficiency_ratio(path: GreedyResult, k_min: int) -> List[int]:
    """Legacy ratio rule: t* = argmax_{t in [k_min, T]} F_t / t."""
    if not path.order:
        return []
    k = max(1, min(int(k_min), len(path.order)))
    seg = path.prefix_efficiency[k - 1:]
    t_star = k + int(np.argmax(seg))
    return path.order[:t_star]


# ------------------- numerical sanity checks (tests) -------------------- #

def is_monotone(obj: Objective, universe: Sequence[int], max_sets: int = 200,
                seed: int = 0) -> bool:
    rng = np.random.default_rng(seed)
    U = [int(x) for x in universe]
    for _ in range(max_sets):
        size = int(rng.integers(0, len(U) + 1))
        S = set(int(x) for x in rng.choice(U, size=size, replace=False)) if size else set()
        j = int(rng.choice(U))
        if obj.value(list(S | {j})) + 1e-9 < obj.value(list(S)):
            return False
    return True


def is_submodular(obj: Objective, universe: Sequence[int], max_checks: int = 300,
                  seed: int = 0) -> bool:
    """Diminishing returns: F(A∪{x}) - F(A) >= F(B∪{x}) - F(B) for A⊆B, x∉B."""
    rng = np.random.default_rng(seed)
    U = [int(x) for x in universe]
    for _ in range(max_checks):
        size_B = int(rng.integers(0, len(U) + 1))
        B = set(int(x) for x in rng.choice(U, size=size_B, replace=False)) if size_B else set()
        size_A = int(rng.integers(0, len(B) + 1)) if B else 0
        A = set(int(x) for x in rng.choice(list(B), size=size_A, replace=False)) if size_A else set()
        rest = [x for x in U if x not in B]
        if not rest:
            continue
        x = int(rng.choice(rest))
        lhs = obj.value(list(A | {x})) - obj.value(list(A))
        rhs = obj.value(list(B | {x})) - obj.value(list(B))
        if lhs + 1e-9 < rhs:
            return False
    return True


def exhaustive_greedy_ratio(obj: Objective, budget: int) -> Tuple[List[int], float]:
    """Tiny-instance optimum by exhaustive enumeration (for ratio recording)."""
    f = len(obj.vertex_mu)
    best_S: List[int] = []
    best_v = -np.inf
    universe = list(range(f))
    from itertools import combinations
    for k in range(0, min(budget, f) + 1):
        for S in combinations(universe, k):
            v = obj.value(list(S))
            if v > best_v:
                best_v, best_S = v, list(S)
    return best_S, float(best_v)
