# Selecting an output from a custom text model

An ONNX graph can expose multiple outputs, such as token hidden states and a pooled
sentence embedding. By default FastEmbed uses the first output, preserving existing
custom model behavior. To select a different output, pass its exact ONNX name when
registering the model:

```python
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType

TextEmbedding.add_custom_model(
    model="my-org/my-embedding-model",
    pooling=PoolingType.DISABLED,
    normalization=True,
    sources=ModelSource(hf="my-org/my-embedding-model"),
    dim=384,
    output_name="sentence_embedding",
)
model = TextEmbedding("my-org/my-embedding-model")
embeddings = list(model.embed(["Example document"]))
```

Use `onnxruntime.InferenceSession(...).get_outputs()` to inspect the graph's output
names. Choose `DISABLED` for an already pooled `[batch, dim]` output, or a token
pooling mode for a `[batch, tokens, dim]` output. Normalization runs after pooling.
The same selection applies to lazy loading and parallel workers. An unknown name
raises ONNX Runtime's normal invalid-output error when inference runs. This selects
one embedding output; it does not return all graph outputs.
