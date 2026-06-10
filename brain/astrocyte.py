"""Neuron-astrocyte associative memory — on the neuron-graph substrate.

After Kozachkov, Slotine & Krotov, "Neuron-Astrocyte Associative Memory"
(PNAS 2025, 122(21):e2417788122; arXiv:2311.08135). The K stored patterns ARE
the astrocytes; retrieval is two passes over a bipartite neuron <-> astrocyte
graph. The N^4 process-coupling tensor T_ijkl is never materialized and no
gradient is used to store memories.

    overlap_mu = sum_i  xi_i^mu * phi(x_i)     # GATHER  (neuron -> astrocyte calcium)
    glio_mu    = Fprime(overlap_mu)            # gliotransmitter nonlinearity
    x_new_i    = sum_mu xi_i^mu * glio_mu       # SCATTER (astrocyte -> neuron)

Two implementations, proven equivalent (see tests/test_astrocyte.py):

- ``NeuronAstrocyteMemory``  — dense numpy reference. Patterns are a (K, N)
  block; gather/scatter are two matmuls. This is the testability ground truth.

- ``SubstrateAstrocyteMemory`` — the substrate wiring. Each astrocyte mu is a
  real neuron in a ``Brain``; its OUTGOING CSR synapse block is its synaptic
  island, one edge ``astrocyte mu -> neuron i`` carrying weight xi_i^mu. Both
  passes walk that edge slice as a sparse multiply-accumulate (no dense matmul,
  no matmul on a critical path — same discipline as spread()). gather/scatter
  are overridden; the dynamics (step/retrieve/glio) are inherited unchanged.

  The process-to-process connectivity ``r = K/N`` from the paper becomes a real,
  measurable knob: it is the size of each astrocyte's neighbor set N(mu) — the
  edge density of the astrocyte<->neuron incidence. r=1 (full) reproduces the
  dense reference bit-for-sign; r<1 (local synaptic islands) is the realized
  capacity the paper points at but does not run. See
  ``brain.tasks.assoc.experiments.density_sweep``.

Interaction regimes (Krotov & Hopfield 2016 energy F(z)=z^n -> K_max ~ N^(n-1)):
  - poly, degree n  ->  Fprime(z)=z^(n-1).  n=2 classic Hopfield (~N),
                        n=4 the paper's quartic (~N^3).
  - softmax, beta   ->  Fprime(z)=softmax(beta*z); the layer reduces to
                        transformer self-attention (paper Appendix C;
                        Ramsauer et al. 2021).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .store import Brain
from .neuron import NeuronType


# --------------------------------------------------------------------------- #
# Dense reference (matmul gather/scatter) — the testability ground truth.
# --------------------------------------------------------------------------- #
class NeuronAstrocyteMemory:
    """Dense numpy associative memory: gather -> F'(.) -> scatter."""

    def __init__(self, interaction: str = "poly", degree: int = 4,
                 beta: float = 1.0, activation: str = "sign"):
        assert interaction in ("poly", "softmax")
        assert activation in ("sign", "identity")
        self.interaction = interaction
        self.degree = degree
        self.beta = beta
        self.activation = activation
        self.patterns: Optional[np.ndarray] = None     # (K, N) astrocyte edge weights

    # ---- storage: one-shot, no backprop -------------------------------------
    def store(self, patterns: np.ndarray) -> "NeuronAstrocyteMemory":
        """Append astrocyte nodes. patterns: (K, N). Pure Hebbian write."""
        patterns = np.asarray(patterns, dtype=np.float64)
        if patterns.ndim == 1:
            patterns = patterns[None, :]
        self.patterns = (patterns if self.patterns is None
                         else np.vstack([self.patterns, patterns]))
        return self

    def reset(self) -> None:
        self.patterns = None

    @property
    def K(self) -> int:
        return 0 if self.patterns is None else self.patterns.shape[0]

    @property
    def N(self) -> int:
        return 0 if self.patterns is None else self.patterns.shape[1]

    # ---- neuron nonlinearity + gliotransmitter ------------------------------
    def _phi(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "sign":
            s = np.sign(x)
            s[s == 0] = 1.0
            return s
        return x

    def _glio(self, overlap: np.ndarray) -> np.ndarray:
        if self.interaction == "poly":
            p = self.degree - 1
            return np.sign(overlap) * np.abs(overlap) ** p     # z^(n-1), sign-safe
        z = self.beta * (overlap - overlap.max())              # softmax / attention
        e = np.exp(z)
        return e / e.sum()

    # ---- the two bipartite passes (dense) -----------------------------------
    def _gather(self, phi: np.ndarray) -> np.ndarray:
        return self.patterns @ phi                              # (K,)

    def _scatter(self, glio: np.ndarray) -> np.ndarray:
        return glio @ self.patterns                             # (N,)

    # ---- dynamics -----------------------------------------------------------
    def step(self, x: np.ndarray) -> np.ndarray:
        """One synchronous gather -> F' -> scatter pass."""
        phi = self._phi(x)
        overlap = self._gather(phi)
        glio = self._glio(overlap)
        out = self._scatter(glio)
        return self._phi(out) if self.activation == "sign" else out

    def retrieve(self, x0: np.ndarray, steps: int = 20) -> np.ndarray:
        """Relax to a fixed point from initial state x0."""
        x = np.array(x0, dtype=np.float64)
        for _ in range(steps):
            xn = self.step(x)
            if self.activation == "sign" and np.array_equal(xn, x):
                break
            x = xn
        return x


# --------------------------------------------------------------------------- #
# Substrate wiring: astrocytes are neurons, gather/scatter are sparse edge MAC.
# --------------------------------------------------------------------------- #
class SubstrateAstrocyteMemory(NeuronAstrocyteMemory):
    """Associative memory built on a ``Brain``.

    N neuron nodes (ids 0..N-1) are edge targets; K astrocyte nodes
    (ids N..N+K-1) each own an outgoing CSR synapse block — its synaptic
    island. Edge ``astrocyte mu -> neuron i`` carries weight xi_i^mu. gather and
    scatter walk that block as a sparse multiply-accumulate; the inherited
    dynamics are unchanged. Only edges in each astrocyte's neighbor set are
    stored, so connectivity ``r`` is a genuine substrate property, not a mask.
    """

    RELATION = "astro"          # label only; gather/scatter read the raw weight

    def __init__(self, n_neurons: int, interaction: str = "poly", degree: int = 4,
                 beta: float = 1.0, activation: str = "sign",
                 brain: Optional[Brain] = None):
        super().__init__(interaction=interaction, degree=degree, beta=beta,
                         activation=activation)
        self.n = int(n_neurons)
        self.brain = brain if brain is not None else Brain()
        # N neuron nodes (ids 0..n-1) as concrete edge targets.
        for i in range(self.n):
            assert self.brain.add_neuron(type=NeuronType.CONCEPT) == i
        if self.RELATION not in self.brain.relation_id:
            self.brain._add_relation(self.RELATION, 1.0)
        self._astro_ids: list[int] = []     # astrocyte neuron ids, in store order

    # ---- storage: append astrocyte neurons with sparse neighbor edges -------
    def store(self, patterns: np.ndarray, *, connectivity: float = 1.0,
              rng: Optional[np.random.Generator] = None) -> "SubstrateAstrocyteMemory":
        """Write each pattern as one astrocyte neuron.

        connectivity: r in (0, 1]. Each astrocyte connects to a random
            round(r*N)-sized neighbor set (r=1 -> full / dense-equivalent).
            Pattern entries outside the neighbor set are simply not stored —
            this is the paper's local synaptic island, the realized-capacity knob.
        """
        patterns = np.asarray(patterns, dtype=np.float64)
        if patterns.ndim == 1:
            patterns = patterns[None, :]
        assert patterns.shape[1] == self.n, \
            f"pattern width {patterns.shape[1]} != n_neurons {self.n}"
        rng = rng if rng is not None else np.random.default_rng()
        n_edges = max(1, min(self.n, int(round(connectivity * self.n))))

        for xi in patterns:
            aid = self.brain.add_neuron(type=NeuronType.ASTROCYTE)
            if n_edges == self.n:
                nbrs = np.arange(self.n)
            else:
                nbrs = rng.choice(self.n, size=n_edges, replace=False)
            edges = [(int(i), self.RELATION, float(xi[i])) for i in nbrs]
            self.brain.set_synapses(aid, edges)
            self._astro_ids.append(aid)
        return self

    def reset(self) -> None:        # pragma: no cover - rebuild is cheaper than mutate
        raise NotImplementedError("rebuild a fresh SubstrateAstrocyteMemory instead")

    @property
    def K(self) -> int:
        return len(self._astro_ids)

    @property
    def N(self) -> int:
        return self.n

    # ---- gather/scatter as sparse CSR edge MAC (overrides dense matmuls) -----
    def _gather(self, phi: np.ndarray) -> np.ndarray:
        """Per astrocyte mu: overlap_mu = sum over its edges of weight * phi[i]."""
        overlap = np.zeros(self.K, dtype=np.float64)
        for mu, aid in enumerate(self._astro_ids):
            e = self.brain.synapses_of(aid)
            if len(e) == 0:
                continue
            idx = e["to_id"].astype(np.intp)
            overlap[mu] = np.dot(e["weight"].astype(np.float64), phi[idx])
        return overlap

    def _scatter(self, glio: np.ndarray) -> np.ndarray:
        """Per astrocyte mu: add weight * glio_mu back onto each neighbor neuron."""
        out = np.zeros(self.n, dtype=np.float64)
        for mu, aid in enumerate(self._astro_ids):
            e = self.brain.synapses_of(aid)
            if len(e) == 0:
                continue
            idx = e["to_id"].astype(np.intp)
            np.add.at(out, idx, e["weight"].astype(np.float64) * glio[mu])
        return out
