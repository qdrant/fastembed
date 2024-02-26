from pathlib import Path
from typing import List

import click
import numpy as np
import torch
import torch.nn.functional as F
from optimum.onnxruntime import ORTModelForFeatureExtraction
from optimum.pipelines import pipeline
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def hf_embed(model_id: str, texts: List[str], tokenizer):
    # Tokenize the input texts
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    model_output = model(**encoded_input)
    sentence_embeddings = model_output[0][:, 0]
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


@click.command()
@click.option("--model_id", help="model id from huggingface.co/models")
@click.option("--model_dir", help="The person to greet.")
def setup(model_id, model_dir):
    text = "This is a test sentence"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    output_dir = Path(model_dir)
    model = ORTModelForFeatureExtraction.from_pretrained(output_dir)
    onnx_quant_embed = pipeline(
        "feature-extraction", model=model, accelerator="ort", tokenizer=tokenizer, return_tensors=True
    )
    quant_embeddings = onnx_quant_embed([text])
    quant_embeddings = F.normalize(quant_embeddings[0][:,0], p=2, dim=1)
    quant_embeddings = quant_embeddings.detach().numpy()
    print(quant_embeddings.shape)

    torch_embeddings = hf_embed(model_id, texts=[text], tokenizer=tokenizer)
    torch_embeddings = F.normalize(torch_embeddings, p=2, dim=1)
    torch_embeddings = torch_embeddings.detach().numpy()
    print(torch_embeddings.shape)
    assert quant_embeddings.shape == torch_embeddings.shape
    print(np.allclose(quant_embeddings, torch_embeddings, atol=1e-5))


if __name__ == "__main__":
    setup()
