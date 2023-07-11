from abc import ABC, abstractmethod


class Embedding(ABC):
    @abstractmethod
    def encode(self, texts):
        pass


class SentenceTransformersEmbedding(Embedding):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install the sentence-transformers package to use this method.")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts)


class OpenAIEmbedding(Embedding):
    def __init__(self):
        # Initialize your OpenAI model here
        # self.model = ...
        ...

    def encode(self, texts):
        # Use your OpenAI model to encode the texts
        # return self.model.encode(texts)
        raise NotImplementedError
