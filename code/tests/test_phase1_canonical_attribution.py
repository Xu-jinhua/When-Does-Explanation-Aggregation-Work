from __future__ import annotations

import numpy as np

from xai_ensemble.phase1._compat_support import canonical_attribution_map


def test_canonical_attribution_map_aligns_rgb_and_attention_channels() -> None:
    rgb = np.asarray(
        [[[[1.0, -1.0]], [[2.0, 1.0]], [[-3.0, 3.0]]]],
        dtype=np.float32,
    )
    attention = np.asarray([[[[0.2, -0.4]]]], dtype=np.float32)

    reduced = canonical_attribution_map(rgb)
    unchanged = canonical_attribution_map(attention)

    assert reduced.shape == attention.shape == (1, 1, 1, 2)
    np.testing.assert_allclose(reduced, np.mean(rgb, axis=1, keepdims=True))
    assert unchanged is attention
