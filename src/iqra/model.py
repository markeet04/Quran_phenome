"""FusedW2VBertForCTC: softmax-weighted fusion of the 24 w2v-BERT encoder layers
before the adapter and CTC head; loads existing HF checkpoints. Implemented in T5."""


class FusedW2VBertForCTC:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("FusedW2VBertForCTC: see SPEC.md")
