# qdrant_client.py


class QdrantClient:
    # Existing QdrantClient implementation...

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Check if fastvector is installed
        try:
            from fastembed.qdrant_mixin import QdrantClientMixin

            # If it is, add the mixin methods to this instance
            for name, method in QdrantClientMixin.__dict__.items():
                if callable(method):
                    setattr(self, name, method.__get__(self, self.__class__))
        except ImportError:
            # If it's not, do nothing
            pass
