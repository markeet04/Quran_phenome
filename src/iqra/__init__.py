"""Detect-then-verify Quranic mispronunciation detection. See SPEC.md."""
import os

# Decision 34: reduce CUDA memory fragmentation for the two-pass consistency loss on a
# 14.56 GiB T4. Must be set before PyTorch first allocates GPU memory; the notebooks'
# install cell imports this package before any model touches the GPU. Only sets the
# variable if the user has not chosen a value.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
