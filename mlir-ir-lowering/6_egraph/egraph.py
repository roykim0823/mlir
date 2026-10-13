"""A minimal e-graph with equality saturation and cost-based extraction.

This is a from-scratch, dependency-free implementation of the idea behind the
`egg`/`egglog` libraries (Part 6 of the reference series uses `egglog`, whose API
moves fast; building the core ourselves keeps the chapter runnable and makes the
algorithm legible).

Key idea: instead of rewriting an expression *destructively* (which forces you to
guess a good rewrite order — the "phase ordering problem"), an **e-graph** stores
*all* equivalent forms at once. You apply every rule until nothing new appears
(**equality saturation**), then **extract** the cheapest form according to a cost
model. Rewrite order stops mattering.

Representation
--------------
* A **term** (and a rewrite **pattern**) is a nested tuple:
      ("mul", ("sym", "a"), ("lit", 2))     # a * 2
  Leaves are ("lit", value) or ("sym", name). A *pattern* may also contain
  `Var("x")` placeholders that match any e-class.
* An **e-node** is (op, tuple-of-child-eclass-ids, data) — an operator applied to
  e-classes, hash-consed so identical nodes are shared.
* An **e-class** is a set of equivalent e-nodes, tracked with union-find.
"""

from itertools import count


class Var:
  """A pattern variable, e.g. Var("x") matches any e-class and binds it."""
  __slots__ = ("name",)

  def __init__(self, name):
    self.name = name


class EGraph:
  def __init__(self):
    self.parent = {}        # union-find: eclass id -> parent id
    self.nodes = {}         # eclass id -> set of e-nodes
    self.hashcons = {}      # e-node -> eclass id (dedup)
    self._ids = count()

  # --- union-find -------------------------------------------------------------

  def _new_class(self):
    i = next(self._ids)
    self.parent[i] = i
    self.nodes[i] = set()
    return i

  def find(self, i):
    while self.parent[i] != i:
      self.parent[i] = self.parent[self.parent[i]]   # path compression
      i = self.parent[i]
    return i

  def _canon(self, enode):
    op, ch, data = enode
    return (op, tuple(self.find(c) for c in ch), data)

  # --- building ---------------------------------------------------------------

  def add(self, term):
    """Insert a term, returning the id of the e-class that represents it."""
    op = term[0]
    if op in ("lit", "sym"):
      enode = (op, (), term[1])
    else:
      enode = (op, tuple(self.add(arg) for arg in term[1:]), None)
    return self._add_enode(enode)

  def _add_enode(self, enode):
    enode = self._canon(enode)
    if enode in self.hashcons:
      return self.find(self.hashcons[enode])
    i = self._new_class()
    self.hashcons[enode] = i
    self.nodes[i].add(enode)
    return i

  def merge(self, a, b):
    """Declare two e-classes equivalent."""
    a, b = self.find(a), self.find(b)
    if a == b:
      return a
    self.parent[b] = a
    self.nodes[a] |= self.nodes[b]
    self.nodes[b] = set()
    return a

  def rebuild(self):
    """Restore congruence: if two e-nodes became identical after merges, their
    e-classes must merge too. Repeat until stable."""
    changed = True
    while changed:
      changed = False
      self.hashcons = {}
      self.nodes = {c: {self._canon(en) for en in s}
                    for c, s in self.nodes.items() if self.find(c) == c}
      for c, s in self.nodes.items():
        for en in s:
          if en in self.hashcons and self.find(self.hashcons[en]) != self.find(c):
            self.merge(self.hashcons[en], c)
            changed = True
          self.hashcons[en] = self.find(c)

  # --- e-matching -------------------------------------------------------------

  def _match(self, pat, eclass, subst):
    """All ways to match `pat` against e-class `eclass`, extending `subst`."""
    eclass = self.find(eclass)
    if isinstance(pat, Var):
      if pat.name in subst:
        return [subst] if subst[pat.name] == eclass else []
      s = dict(subst)
      s[pat.name] = eclass
      return [s]
    out = []
    for op, ch, data in self.nodes[eclass]:
      if pat[0] in ("lit", "sym"):
        if op == pat[0] and data == pat[1]:
          out.append(subst)
        continue
      if pat[0] != op or len(pat) - 1 != len(ch):
        continue
      results = [subst]
      for p, c in zip(pat[1:], ch):
        results = [s2 for s in results for s2 in self._match(p, c, s)]
      out += results
    return out

  def ematch(self, pat):
    """Find every (eclass, substitution) where `pat` matches in the graph."""
    res = []
    for c in list(self.parent):
      if self.find(c) == c:
        for sb in self._match(pat, c, {}):
          res.append((c, sb))
    return res

  def _instantiate(self, rhs, subst):
    if isinstance(rhs, Var):
      return subst[rhs.name]            # already an e-class id
    if callable(rhs):
      return rhs(self, subst)           # dynamic right-hand side (e.g. constant fold)
    if rhs[0] in ("lit", "sym"):
      return self._add_enode((rhs[0], (), rhs[1]))
    ch = tuple(self._instantiate(a, subst) for a in rhs[1:])
    return self._add_enode((rhs[0], ch, None))

  # --- the main loop ----------------------------------------------------------

  def saturate(self, rules, max_iters=30):
    """Apply all rules until no new equivalences appear (or the cap is hit)."""
    for _ in range(max_iters):
      matches = [(c, rhs, sb)
                 for lhs, rhs in rules
                 for c, sb in self.ematch(lhs)]
      sig_before = (len(self.hashcons),
                    sum(len(s) for s in self.nodes.values()))
      for c, rhs, sb in matches:
        self.merge(c, self._instantiate(rhs, sb))
      self.rebuild()
      sig_after = (len(self.hashcons),
                   sum(len(s) for s in self.nodes.values()))
      if sig_after == sig_before:
        break

  def extract(self, eclass, costs):
    """Return the lowest-cost term in `eclass`, given per-op `costs`.

    cost(node) = costs[op] + sum of best child costs; best(eclass) = cheapest
    node. Iterated to a fixpoint so cyclic equivalences converge."""
    best = {}
    for _ in range(200):
      changed = False
      for c in list(self.parent):
        if self.find(c) != c:
          continue
        for op, ch, data in self.nodes[c]:
          if any(self.find(x) not in best for x in ch):
            continue
          cost = costs.get(op, 1) + sum(best[self.find(x)][0] for x in ch)
          if c not in best or cost < best[c][0]:
            term = ((op, data) if op in ("lit", "sym")
                    else (op,) + tuple(best[self.find(x)][1] for x in ch))
            best[c] = (cost, term)
            changed = True
      if not changed:
        break
    return best[self.find(eclass)][1]
