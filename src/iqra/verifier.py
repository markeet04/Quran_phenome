"""LightGBM flag verifier: train, grouped CV, predict, robust tau sweep.
Implemented in T3."""


def train_verifier(*args, **kwargs):
    raise NotImplementedError("train_verifier: see SPEC.md")


def sweep_tau(*args, **kwargs):
    raise NotImplementedError("sweep_tau: see SPEC.md")
