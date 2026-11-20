"""Posterior caches: clip ID -> float16 log-posteriors [T x V], greedy tokens,
emission spans; vocab hash and frame rate stored and checked. Implemented in T2."""


def write_cache(*args, **kwargs):
    raise NotImplementedError("write_cache: see SPEC.md")


def read_cache(*args, **kwargs):
    raise NotImplementedError("read_cache: see SPEC.md")
