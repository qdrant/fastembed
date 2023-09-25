# ⚡️ What is FastEmbed?

FastEmbed is an easy to use -- lightweight, fast, Python library built for retrieval embedding generation. 

The default embedding supports "query" and "passage" prefixes for the input text. The default model is Flag Embedding, which is top of the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard. Here is an example for [Retrieval Embedding Generation](https://qdrant.github.io/fastembed/examples/Retrieval%20with%20FastEmbed/)

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
embeddings: List[np.ndarray] = list(embedding_model.embed(documents))
```

### Why fast?

It's important we justify the "fast" in FastEmbed. FastEmbed is fast because:

1. Quantized model weights
2. ONNX Runtime which allows for fast inference on CPU and other dedicated runtimes

### Why light?
1. No hidden dependencies on PyTorch or TensorFlow via Huggingface Transformers
2. We do use the tokenizer from Huggingface Transformers, but it's a light dependency

### Why accurate?
1. Better than OpenAI Ada-002
2. Top of the Embedding leaderboards e.g. [MTEB](https://huggingface.co/spaces/mteb/leaderboard)

#### Similar Work
Ilyas M. wrote about using [FlagEmbeddings with Optimum](https://twitter.com/IlysMoutawwakil/status/1705215192425288017) over CUDA.