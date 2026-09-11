# External RelProp provider

Phase 1 uses Hila Chefer's `Transformer-Explainability` implementation for
genuine ViT `LRP`, `PartialLRP`, and `FullLRP`.  The minimal transitive source
snapshot needed by these methods is vendored at the immutable revision
`c3e578f76b954e8528afeaaee26de3f07e3fe559`.  Every imported source file is
verified against a SHA-256 allowlist before model construction.

From `code/`, with `XAI_PYTHON` and `PYTHONPATH` configured as in the main
README, verify the tracked snapshot with:

```bash
PYTHONPATH=src "$XAI_PYTHON" scripts/setup_transformer_explainability.py --verify-only
```

The tracked default source is `code/vendor/Transformer-Explainability`, so new
Git checkouts and worktrees need no external setup.  The optional
`XAI_TRANSFORMER_EXPLAINABILITY_ROOT` override can point development runs to a
complete external checkout.  A wrong revision, changed required file,
unsupported architecture, or non-strict Phase-0 checkpoint fails closed.  The
GPU explainer compatibility pilot additionally compares logits and predictions
against the timm model before any RelProp method can enter the formal roster.

The upstream project is MIT licensed.  Its license notice is retained both in
`Transformer-Explainability.LICENSE` and inside the tracked snapshot as
`Transformer-Explainability/LICENSE`; the latter is checksum-verified with the
runtime sources.
