from fastembed import (
    TextEmbedding,
    SparseTextEmbedding,
    ImageEmbedding,
    LateInteractionMultimodalEmbedding,
    LateInteractionTextEmbedding,
)


def test_text_list_supported_models():
    for model_type in [
        TextEmbedding,
        SparseTextEmbedding,
        ImageEmbedding,
        LateInteractionMultimodalEmbedding,
        LateInteractionTextEmbedding,
    ]:
        supported_models = model_type.list_supported_models()
        assert isinstance(supported_models, list)
        description = supported_models[0]
        assert isinstance(description, dict)

        assert "model" in description and description["model"]
        if model_type != SparseTextEmbedding:
            assert "dim" in description and description["dim"]
        assert "license" in description and description["license"]
        assert "size_in_GB" in description and description["size_in_GB"]
        assert "model_file" in description and description["model_file"]
        assert "sources" in description and description["sources"]
        assert "hf" in description["sources"] or "url" in description["sources"]
