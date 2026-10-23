# Local embedding server

Install the optional server dependencies and start a text embedding model.

```bash
pip install 'fastembed[server]'
fastembed serve --model-name BAAI/bge-small-en-v1.5
```

`python -m fastembed serve` starts the same server. The default address is
`127.0.0.1` and the default port is `8080`. The model is loaded once before
the server starts. Omitting `--model-name` uses the library's default model.

Use `--model-path /path/to/model-directory` to load existing model files without
downloading them. The directory must contain the model and tokenizer files
expected by the selected supported model. A standalone ONNX file is insufficient.
The model name still selects the architecture and preprocessing.

`--device cpu` is the default. `--device cuda` requires an ONNX Runtime installation
with the CUDA execution provider and the corresponding CUDA libraries.
`--cache-dir`, `--threads` and `--batch-size` control the model cache, ONNX Runtime
threads and inference batch size. `fastembed serve --help` lists all options.

## Requests

```bash
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"BAAI/bge-small-en-v1.5","input":["A local embedding","Another sentence"]}'
```

The response contains an ordered `data` list with an embedding and index for each
input, the loaded model name and token usage from the model's tokenizer after
truncation, including special tokens.
`GET /v1/models` lists the loaded model and `GET /health` reports server readiness.

The endpoint accepts a nonempty string or a list of up to 2048 nonempty strings.
Token ID inputs are unsupported. Text is tokenized and truncated using the model's
normal settings. `encoding_format` accepts `float` or `base64`. `dimensions` may
only specify the model's native embedding size. An omitted `model` uses the loaded
model, and requesting a different model returns an error without loading it.
Requests run through `embed`, with the same preprocessing as the Python library.

An OpenAI client can use the local endpoint with a placeholder key.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
result = client.embeddings.create(
    model="BAAI/bge-small-en-v1.5", input=["A local embedding"]
)
print(result.data[0].embedding)
```

The server does not implement authentication. Keep the default loopback binding
for local use. To share it, configure access control with a reverse proxy before
binding to an external interface with `--host`. Inference uses a worker thread
and is serialized per model to avoid concurrent tokenizer mutation. Disconnecting
a client does not cancel queued or running inference.
