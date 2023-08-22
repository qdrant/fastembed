# 🪶 What is FastEmbed?

FastEmbed is lightweight, fast, Python library built for retrieval and easy to use. 


## 🚀 Installation

To install the FastEmbed library, pip works: 

```bash
pip install fastembed
```

## 📖 Usage

```python
from fastembed.embedding import DefaultEmbedding

documents: List[str] = [
    "Hello, World!",
    "This is an example document.",
    "fastembed is supported by and maintained by Qdrant." * 128,
]
embedding_model = DeafultEmbedding() 
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