# FastEmbed Library

FastEmbed optimises for being Accurate and Fast:

1. Accuracy/Recall
    - Better than OpenAI Ada-002
    - Top of the leaderboard from [MTEB](https://huggingface.co/spaces/mteb/leaderboard)

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
