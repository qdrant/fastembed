class GeneralTextEmbedding(Embedding):
    """
    https://huggingface.co/thenlper/gte-large

    SoTA embedding model for text based retrieval tasks.
    """

    @classmethod
    def average_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def __init__(self, model_name="thenlper/gte-large"):
        try:
            import torch.nn.functional as F
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("Please install the transformers package with torch to use this method.")
        self.tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
        self.model = AutoModel.from_pretrained("thenlper/gte-large")

    def encode(self, input_texts: List[str]):
        try:
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("Please install Pytorch to use this method.")
        # Tokenize the input texts
        batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt")

        outputs = self.model(**batch_dict)
        embeddings = GeneralTextEmbedding.average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

        # (Optionally) normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:1] @ embeddings[1:].T) * 100
        return scores