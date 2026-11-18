# ⚡️ What is FastEmbed?

FastEmbed is a lightweight, fast, Python library built for embedding generation. We [support popular text models](https://qdrant.github.io/fastembed/examples/Supported_Models/). Please [open a GitHub issue](https://github.com/qdrant/fastembed/issues/new) if you want us to add a new model.

The default text embedding (`TextEmbedding`) model is Flag Embedding, presented in the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard. It supports "query" and "passage" prefixes for the input text. Here is an example for [Retrieval Embedding Generation](https://qdrant.github.io/fastembed/qdrant/Retrieval_with_FastEmbed/) and how to use [FastEmbed with Qdrant](https://qdrant.github.io/fastembed/qdrant/Usage_With_Qdrant/).

## 📈 Why FastEmbed?

1. Light: FastEmbed is a lightweight library with few external dependencies. We don't require a GPU and don't download GBs of PyTorch dependencies, and instead use the ONNX Runtime. This makes it a great candidate for serverless runtimes like AWS Lambda. 

2. Fast: FastEmbed is designed for speed. We use the ONNX Runtime, which is faster than PyTorch. We also use data parallelism for encoding large datasets.

3. Accurate: FastEmbed is better than OpenAI Ada-002. We also [support](https://qdrant.github.io/fastembed/examples/Supported_Models/) an ever-expanding set of models, including a few multilingual models.

## 🚀 Installation

To install the FastEmbed library, pip works best. You can install it with or without GPU support:

```bash
pip install fastembed

# or with GPU support

pip install fastembed-gpu
```

## 📖 Quickstart

```python
from fastembed import TextEmbedding


# Example list of documents
documents: list[str] = [
    "This is built to be faster and lighter than other embedding libraries e.g. Transformers, Sentence-Transformers, etc.",
    "fastembed is supported by and maintained by Qdrant.",
]

# This will trigger the model download and initialization
embedding_model = TextEmbedding()
print("The model BAAI/bge-small-en-v1.5 is ready to use.")

embeddings_generator = embedding_model.embed(documents)  # reminder this is a generator
embeddings_list = list(embedding_model.embed(documents))
  # you can also convert the generator to a list, and that to a numpy array
len(embeddings_list[0]) # Vector of 384 dimensions
```

Fastembed supports a variety of models for different tasks and modalities.
The list of all the available models can be found [here](https://qdrant.github.io/fastembed/examples/Supported_Models/)
### 🎒 Dense text embeddings

```python
from fastembed import TextEmbedding

model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
embeddings = list(model.embed(documents))

# [
#   array([-0.1115,  0.0097,  0.0052,  0.0195, ...], dtype=float32),
#   array([-0.1019,  0.0635, -0.0332,  0.0522, ...], dtype=float32)
# ]

```

Dense text embedding can also be extended with models which are not in the list of supported models.

```python
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource

TextEmbedding.add_custom_model(
    model="intfloat/multilingual-e5-small",
    pooling=PoolingType.MEAN,
    normalization=True,
    sources=ModelSource(hf="intfloat/multilingual-e5-small"),  # can be used with an `url` to load files from a private storage
    dim=384,
    model_file="onnx/model.onnx",  # can be used to load an already supported model with another optimization or quantization, e.g. onnx/model_O4.onnx
)
model = TextEmbedding(model_name="intfloat/multilingual-e5-small")
embeddings = list(model.embed(documents))
```


### 🔱 Sparse text embeddings

* SPLADE++

```python
from fastembed import SparseTextEmbedding

model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
embeddings = list(model.embed(documents))

# [
#   SparseEmbedding(indices=[ 17, 123, 919, ... ], values=[0.71, 0.22, 0.39, ...]),
#   SparseEmbedding(indices=[ 38,  12,  91, ... ], values=[0.11, 0.22, 0.39, ...])
# ]
```

For MiniCOIL, use `max_sequence_length` to cap the tokens sent to the model when
embedding long texts. The limit includes special tokens and must leave room for at least one
text token. Values above the model's own tokenizer limit are capped at that limit.

```python
model = SparseTextEmbedding(model_name="Qdrant/minicoil-v1", max_sequence_length=512)
embeddings = list(model.embed(documents))
```

<!--
* BM42 - ([link](ToDo))

```
from fastembed import SparseTextEmbedding

model = SparseTextEmbedding(model_name="Qdrant/bm42-all-minilm-l6-v2-attentions")
embeddings = list(model.embed(documents))

# [
#   SparseEmbedding(indices=[ 17, 123, 919, ... ], values=[0.71, 0.22, 0.39, ...]),
#   SparseEmbedding(indices=[ 38,  12,  91, ... ], values=[0.11, 0.22, 0.39, ...])
# ]
```
-->

### 🦥 Late interaction models (aka ColBERT)


```python
from fastembed import LateInteractionTextEmbedding

model = LateInteractionTextEmbedding(model_name="colbert-ir/colbertv2.0")
embeddings = list(model.embed(documents))

# [
#   array([
#       [-0.1115,  0.0097,  0.0052,  0.0195, ...],
#       [-0.1019,  0.0635, -0.0332,  0.0522, ...],
#   ]),
#   array([
#       [-0.9019,  0.0335, -0.0032,  0.0991, ...],
#       [-0.2115,  0.8097,  0.1052,  0.0195, ...],
#   ]),  
# ]
```

### 🖼️ Image embeddings

```python
from fastembed import ImageEmbedding

images = [
    "./path/to/image1.jpg",
    "./path/to/image2.jpg",
]

model = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
embeddings = list(model.embed(images))

# [
#   array([-0.1115,  0.0097,  0.0052,  0.0195, ...], dtype=float32),
#   array([-0.1019,  0.0635, -0.0332,  0.0522, ...], dtype=float32)
# ]
```

### Late interaction multimodal models (ColPali)

```python
from fastembed import LateInteractionMultimodalEmbedding

doc_images = [
    "./path/to/qdrant_pdf_doc_1_screenshot.jpg",
    "./path/to/colpali_pdf_doc_2_screenshot.jpg",
]

query = "What is Qdrant?"

model = LateInteractionMultimodalEmbedding(model_name="Qdrant/colpali-v1.3-fp16")
doc_images_embeddings = list(model.embed_image(doc_images))
# shape (2, 1030, 128)
# [array([[-0.03354, -0.0209, ..., -0.1558, -0.0768]], dtype=float16)]
query_embedding = model.embed_text(query)
# shape (1, 20, 128)
# [array([[-0.002182, 0.1476, ..., -0.02208, 0.1683]], dtype=float16)]
```

### 🔄 Rerankers
```python
from fastembed.rerank.cross_encoder import TextCrossEncoder

query = "Who is maintaining Qdrant?"
documents: list[str] = [
    "This is built to be faster and lighter than other embedding libraries e.g. Transformers, Sentence-Transformers, etc.",
    "fastembed is supported by and maintained by Qdrant.",
]
encoder = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-6-v2")
scores = list(encoder.rerank(query, documents))

# [-11.48061752319336, 5.472434997558594]
```

Text cross encoders can also be extended with models which are not in the list of supported models.

```python
from fastembed.rerank.cross_encoder import TextCrossEncoder 
from fastembed.common.model_description import ModelSource

TextCrossEncoder.add_custom_model(
    model="Xenova/ms-marco-MiniLM-L-4-v2",
    model_file="onnx/model.onnx",
    sources=ModelSource(hf="Xenova/ms-marco-MiniLM-L-4-v2"),
)
model = TextCrossEncoder(model_name="Xenova/ms-marco-MiniLM-L-4-v2")
scores = list(model.rerank_pairs(
    [("What is AI?", "Artificial intelligence is ..."), ("What is ML?", "Machine learning is ..."),]
))
```

## ⚡️ FastEmbed on a GPU

FastEmbed supports running on GPU devices.
It requires installation of the `fastembed-gpu` package.

```bash
pip install fastembed-gpu
```

Check our [example](https://qdrant.github.io/fastembed/examples/FastEmbed_GPU/) for detailed instructions, CUDA 12.x support and troubleshooting of the common issues.

```python
from fastembed import TextEmbedding

embedding_model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5", 
    providers=["CUDAExecutionProvider"]
)
print("The model BAAI/bge-small-en-v1.5 is ready to use on a GPU.")

```

## Usage with Qdrant

Installation with Qdrant Client in Python:

```bash
pip install qdrant-client[fastembed]
```

or 

```bash
pip install qdrant-client[fastembed-gpu]
```

You might have to use quotes ```pip install 'qdrant-client[fastembed]'``` on zsh.

```python
from qdrant_client import QdrantClient, models

# Initialize the client
client = QdrantClient("localhost", port=6333) # For production
# client = QdrantClient(":memory:") # For experimentation

model_name = "sentence-transformers/all-MiniLM-L6-v2"
payload = [
    {"document": "Qdrant has Langchain integrations", "source": "Langchain-docs", },
    {"document": "Qdrant also has Llama Index integrations", "source": "LlamaIndex-docs"},
]
docs = [models.Document(text=data["document"], model=model_name) for data in payload]
ids = [42, 2]

client.create_collection(
    "demo_collection",
    vectors_config=models.VectorParams(
        size=client.get_embedding_size(model_name), distance=models.Distance.COSINE)
)

client.upload_collection(
    collection_name="demo_collection",
    vectors=docs,
    ids=ids,
    payload=payload,
)

search_result = client.query_points(
    collection_name="demo_collection",
    query=models.Document(text="This is a query document", model=model_name)
).points
print(search_result)
```
