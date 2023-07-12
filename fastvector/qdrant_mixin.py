import uuid
from typing import Any, Dict, Generator, List, Optional
from warnings import warn

from pydantic import BaseModel
from qdrant_client.models import PointStruct, VectorParams, SearchParams, Distance

from .embedding import Embedding, SentenceTransformersEmbedding


class QueryResponse(BaseModel):
    ids: List[str]
    embeddings: List[List[float]]
    metadatas: List[Dict[str, Any]]
    distances: List[float]


class QdrantClientMixin:
    def batch_iterable(self, iterable: List[Any], batch_size: int) -> Generator[List[Any], None, None]:
        """A generator that yields batches of items from an iterable."""
        batch = []
        for item in iterable:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def create_collection(
        self,
        collection_name: str,
        vectors_config: VectorParams = VectorParams(size=1536, distance=Distance.COSINE),
        **kwargs,
    ) -> None:
        """
        Create a collection with the given name and vectors config.

        Args:
            collection_name (str): _description_
            vectors_config (VectorParams, optional): _description_. Defaults to VectorParams(size=1536, distance=Distance.COSINE).
        """
        self.recreate_collection(collection_name=collection_name, vectors_config=vectors_config, **kwargs)

    def upsert_docs(
        self,
        collection_name: str,
        docs: Dict[str, List[Any]],
        batch_size: int = 512,
        wait: bool = True,
        embedding_model: Embedding = SentenceTransformersEmbedding(),
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
                self.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=len(embeddings[0]), distance=Distance.COSINE),
                )
            # Call the existing upsert method with the new PointStruct
            self.upsert(collection_name=collection_name, points=points, wait=wait, **kwargs)

    def search_docs(
        self,
        collection_name: str,
        query_texts: List[str],
        n_results: int = 2,
        batch_size: int = 512,
        query_filter: Optional[Dict[str, Any]] = None,
        search_params: SearchParams = SearchParams(hnsw_ef=128, exact=False),
        embedding_model: Embedding = SentenceTransformersEmbedding(),
        **kwargs,
    ) -> List[QueryResponse]:
        """
        Search for documents in a collection.

        Args:
            collection_name (str): _description_
            query_texts (List[str]): _description_
            n_results (int, optional): _description_. Defaults to 2.
            query_filter (Optional[Dict[str, Any]], optional): _description_. Defaults to None.
            search_params (models.SearchParams, optional): _description_. Defaults to models.SearchParams(hnsw_ef=128, exact=False).
            embedding_model (Embedding, optional): _description_. Defaults to SentenceTransformersEmbedding().
            batch_size (int, optional): _description_. Defaults to 512.

        Returns:
            List[QueryResponse]: _description_
        """

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

    # Define aliases for the methods
    add = upsert_docs
    query = search_docs
