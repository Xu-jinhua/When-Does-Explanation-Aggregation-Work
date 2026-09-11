from __future__ import annotations

import pytest

from xai_ensemble.phase1.transformer import TransformerAttentionExplainer


def test_attention_gradcam_matches_the_last_layer_head_weight_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    attention = (torch.arange(50, dtype=torch.float32).reshape(1, 2, 5, 5) / 10.0).requires_grad_(
        True
    )
    class_zero_gradient = torch.zeros_like(attention)
    class_zero_gradient[:, 0, 0, 1:] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    class_zero_gradient[:, 1, 0, 1:] = torch.tensor([-1.0, 1.0, 3.0, 5.0])
    class_one_gradient = torch.zeros_like(attention)
    class_one_gradient[:, 0, 0, 1:] = torch.tensor([4.0, 2.0, 0.0, -2.0])
    class_one_gradient[:, 1, 0, 1:] = torch.tensor([2.0, 0.0, -2.0, -4.0])

    class Model:
        def __init__(self) -> None:
            self.attention = attention

        def __call__(self, _inputs):
            return torch.stack(
                (
                    (self.attention * class_zero_gradient).sum(),
                    (self.attention * class_one_gradient).sum(),
                )
            ).reshape(1, 2)

        def zero_grad(self, *, set_to_none: bool = True) -> None:
            assert set_to_none
            self.attention.grad = None

    class Capture:
        def __init__(self, model, *, retain_grad: bool) -> None:
            assert retain_grad
            self.model = model
            self.attentions = []

        def __enter__(self):
            self.model.attention.retain_grad()
            self.attentions = [self.model.attention]
            return self

        def __exit__(self, *_args) -> None:
            return None

    monkeypatch.setattr("xai_ensemble.phase1.transformer.AttentionCapture", Capture)
    model = Model()
    explainer = TransformerAttentionExplainer(model, "AttentionGradCAM")
    inputs = torch.zeros(1, 3, 4, 4)

    observed_zero = explainer.attribute(inputs, target=torch.tensor([0]))
    observed_one = explainer.attribute(inputs, target=torch.tensor([1]))

    def expected(gradient):
        cls_patch_attention = attention.detach()[:, :, 0, 1:]
        head_weights = gradient[:, :, 0, 1:].mean(dim=-1, keepdim=True)
        return (cls_patch_attention * head_weights).mean(dim=1).clamp_min(0).reshape(1, 1, 2, 2)

    torch.testing.assert_close(observed_zero, expected(class_zero_gradient))
    torch.testing.assert_close(observed_one, expected(class_one_gradient))
    assert not torch.equal(observed_zero, observed_one)
