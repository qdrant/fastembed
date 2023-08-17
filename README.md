# FastEmbed Library

FastEmbed is light, built for retrieval and fast:

0. Light
    - Quantized model weights
    - ONNX Runtime for inference
    - No hidden dependencies on PyTorch or TensorFlow via Huggingface Transformers

1. Accuracy/Recall
    - Better than OpenAI Ada-002
    - Default is Flag Embedding, which is top of the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard

2. Fast
    - About 2x faster than Huggingface (PyTorch) transformers on single queries
    - Lot faster for batches!
    - ONNX Runtime allows you to use dedicated runtimes for even higher throughput and lower latency 

## Installation

To install the FastEmbed library, we recommend using Poetry, alternatively -- pip works: 

```bash
pip install fastembed
```

## Usage

```python
from fastembed.embedding import DefaultEmbedding

documents: List[str] = [
    "Hello, World!",
    "This is an example document.",
    "fastembed is supported by and maintained by Qdrant." * 128,
]
# Initialize the DefaultEmbedding class with the desired parameters
# model_name="BAAI/bge-small-en"
embedding_model = DeafultEmbedding() 
embeddings: List[np.ndarray] = list(embedding_model.encode(documents))
```
