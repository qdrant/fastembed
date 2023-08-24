# ⚡️ What is FastEmbed?

FastEmbed is an easy to use -- lightweight, fast, Python library built for retrieval embedding generation. 

The default embedding supports "query" and "passage" prefixes for the input text. The default model is [Flag Embedding](https://github.com/FlagOpen/FlagEmbedding), which is top of the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard.

To install the FastEmbed library, pip works: 

```bash
pip install fastembed
```

## 📖 Usage

```python
from fastembed.embedding import DefaultEmbedding

documents: List[str] = [
    "passage: Hello, World!",
    "query: Hello, World!", # these are two different embedding
    "passage: This is an example passage.",
    # You can leave out the prefix but it's recommended
    "fastembed is supported by and maintained by Qdrant." 
]
embedding_model = DefaultEmbedding() 
embeddings: List[np.ndarray] = list(embedding_model.embed(documents))
```

## 🚒 Under the hood

### Why fast?

It's important we justify the "fast" in FastEmbed. FastEmbed is fast because:

1. Quantized model weights
2. ONNX Runtime which allows for inference on CPU, GPU, and other dedicated runtimes

### Why light?
1. No hidden dependencies on PyTorch or TensorFlow via Huggingface Transformers

### Why accurate?
1. Better than OpenAI Ada-002
2. Top of the Embedding leaderboards e.g. [MTEB](https://huggingface.co/spaces/mteb/leaderboard)