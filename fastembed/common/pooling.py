import numpy as np
from numpy import ufunc


class LateInteractionPooler(object):
    def __init__(self, agg_type="row", agg="mean"):
        self.agg = agg
        self.type = agg_type

    def _pick_operation(self) -> ufunc:
        if self.agg == "mean":
            return np.mean
        elif self.agg == "max":
            return np.max
        elif self.agg == "min":
            return np.min
        else:
            raise NotImplementedError(
                f"LateInteractionPooler only supports agg=mean,min,max, provided {self.agg}"
            )

    def pool(self, embeddings_batch) -> np.array:
        if isinstance(embeddings_batch, np.ndarray) and len(embeddings_batch.shape) == 2:
            embeddings_batch = [embeddings_batch]

        if self.type == "row":
            pooled_embedding = self.pool_row(embeddings_batch)
        elif self.type == "col":
            pooled_embedding = self.pool_col(embeddings_batch)
        else:
            raise ValueError("type must be 'row' or 'col'")
        return pooled_embedding

    def pool_row(self, embeddings_batch) -> np.array:
        return self._pick_operation()(embeddings_batch, axis=-1)

    def pool_col(self, embeddings_batch) -> np.array:
        return self._pick_operation()(embeddings_batch, axis=-2)
