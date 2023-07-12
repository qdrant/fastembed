import uuid
from typing import Any, Dict, Generator, List, Optional
from warnings import warn

from pydantic import BaseModel
from qdrant_client.http import models
from qdrant_client.models import PointStruct

from .embedding import Embedding, SentenceTransformersEmbedding


class QueryResponse(BaseModel):
    ids: List[str]
    embeddings: List[List[float]]
    metadatas: List[Dict[str, Any]]
    distances: List[float]


class QdrantClientMixin:
    def batch_iterable(self, iterable: List[Any], batch_size: int = 512) -> Generator[List[Any], None, None]:
        """A generator that yields batches of items from an iterable."""
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def upsert_docs(
        self,
        collection_name: str,
        docs: Dict[str, List[Any]],
        batch_size: int = 512,
        wait: bool = True,
        embedding_model: Embedding = SentenceTransformersEmbedding,
        **kwargs,
    ) -> None:
        """
        Args:
            collection_name (str): _description_
            docs (Dict[str, List[Any]]): _description_
            batch_size (int, optional): _description_. Defaults to 512.
            wait (bool, optional): _description_. Defaults to True.
            embedding_model (Embedding, optional): Defaults to SentenceTransformersEmbedding.
        """

        # Iterate over documents and metadatas in batches
        for batch_docs, batch_metadatas in zip(
            self.batch_iterable(docs["documents"], batch_size),
            self.batch_iterable(docs["metadatas"], batch_size),
        ):
            # Tokenize, embed, and index each document
            embeddings = embedding_model.encode(batch_docs)

            # Create a PointStruct for each document
            points = [
                PointStruct(id=uuid.uuid4().hex, vector=embedding.tolist(), payload={**metadata, "text": doc})
                for doc, embedding, metadata in zip(batch_docs, embeddings, batch_metadatas)
            ]

            # Check if collection exists
            if collection_name not in self.get_collections():
                warn(f"Collection {collection_name} not found. Creating it.")
                self.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(size=len(embeddings[0]), distance=models.Distance.COSINE),
                )
            # Call the existing upsert method with the new PointStruct
            self.upsert(collection_name=collection_name, points=points, wait=wait, **kwargs)

    def search_docs(
        self,
        collection_name: str,
        query_texts: List[str],
        n_results: int = 2,
        query_filter: Optional[Dict[str, Any]] = None,
        search_params: Optional[models.SearchParams] = None,
        embedding_model: Optional[Embedding] = None,
        batch_size: int = 512,
        **kwargs,
    ) -> List[QueryResponse]:
        # If no embedding model is provided, use SentenceTransformersEmbedding by default
        if embedding_model is None:
            embedding_model = SentenceTransformersEmbedding()

        # Define default search parameters if not provided
        if search_params is None:
            search_params = models.SearchParams(hnsw_ef=128, exact=False)

        query_responses = []

        # Perform the search for each batch of query texts
        for query_texts_batch in self.batch_iterable(query_texts, batch_size):
            query_vectors = embedding_model.encode(query_texts_batch)

            for _, query_vector in zip(query_texts_batch, query_vectors):
                search_result = self.search(
                    collection_name=collection_name,
                    query_filter=query_filter,
                    search_params=search_params,
                    query_vector=query_vector.tolist(),
                    limit=n_results,
                    **kwargs,
                )

                ids, embeddings, metadatas, distances = [], [], [], []
                for scored_point in search_result:
                    ids.append(scored_point.id)
                    embeddings.append(scored_point.vector)
                    metadatas.append(scored_point.payload)
                    distances.append(scored_point.score)

                query_responses.append(
                    QueryResponse(
                        ids=ids,
                        embeddings=embeddings,
                        metadatas=metadatas,
                        distances=distances,
                    )
                )

        return query_responses
