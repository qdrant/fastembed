# ⚡️ What is FastEmbed?

FastEmbed is an easy to use -- lightweight, fast, Python library built for retrieval augmented generation. The default embedding supports "query" and "passage" prefixes for the input text.

1. Light
    - Quantized model weights
    - ONNX Runtime for inference
    - No hidden dependencies on PyTorch or TensorFlow via Huggingface Transformers

2. Accuracy/Recall
    - Better than OpenAI Ada-002
    - Default is Flag Embedding, which is top of the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard

3. Fast
    - About 2x faster than Huggingface (PyTorch) transformers on single queries
    - Lot faster for batches!
    - ONNX Runtime allows you to use dedicated runtimes for even higher throughput and lower latency 

## 🚀 Installation

To install the FastEmbed library, pip works: 

```bash
pip install fastembed
```

## 📖 Usage

```python
from fastembed.embedding import FlagEmbedding as Embedding

documents: List[str] = [
    "passage: Hello, World!",
    "query: Hello, World!", # these are two different embedding
    "passage: This is an example passage.",
    # You can leave out the prefix but it's recommended
    "fastembed is supported by and maintained by Qdrant." 
]
embedding_model = Embedding(model_name="BAAI/bge-base-en", max_length=512) 
embeddings: List[np.ndarray] = list(embedding_model.encode(documents))
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