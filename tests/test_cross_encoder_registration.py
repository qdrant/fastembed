"""Keep custom cross-encoder registration consistent with model lookup."""

from pathlib import Path

import pytest

from fastembed.common.model_description import ModelSource
from fastembed.rerank.cross_encoder import TextCrossEncoder
from fastembed.rerank.cross_encoder.custom_text_cross_encoder import CustomTextCrossEncoder
from fastembed.rerank.cross_encoder.onnx_text_cross_encoder import OnnxTextCrossEncoder


@pytest.mark.parametrize("existing", ["builtin", "custom"])
@pytest.mark.parametrize("case", [str, str.lower, str.upper, str.swapcase])
def test_reject_duplicate_names_regardless_of_case(monkeypatch, existing: str, case) -> None:
    """Names resolving to an existing model must not register an unreachable replacement."""
    monkeypatch.setattr(CustomTextCrossEncoder, "SUPPORTED_MODELS", [])
    if existing == "builtin":
        name = OnnxTextCrossEncoder._list_supported_models()[0].model
    else:
        name = "example/MyReranker"
        TextCrossEncoder.add_custom_model(name, sources=ModelSource(hf="example/original"))
    before = TextCrossEncoder.list_supported_models()

    with pytest.raises(ValueError, match="already registered"):
        TextCrossEncoder.add_custom_model(case(name), sources=ModelSource(hf="example/other"))

    assert TextCrossEncoder.list_supported_models() == before


def test_distinct_custom_models_remain_available(monkeypatch, tmp_path: Path) -> None:
    """Construct each distinct model using a case variant without loading ONNX weights."""
    monkeypatch.setattr(CustomTextCrossEncoder, "SUPPORTED_MODELS", [])
    for name in ("example/FirstReranker", "example/SecondReranker"):
        TextCrossEncoder.add_custom_model(name, sources=ModelSource(hf=name))
        encoder = TextCrossEncoder(
            name.upper(),
            cache_dir=str(tmp_path),
            specific_model_path=str(tmp_path),
            lazy_load=True,
            local_files_only=True,
            cuda=False,
        )
        assert isinstance(encoder.model, CustomTextCrossEncoder)
        assert encoder.model.model_description.sources.hf == name

    assert len(CustomTextCrossEncoder.SUPPORTED_MODELS) == 2
