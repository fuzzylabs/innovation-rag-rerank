"""Embedding using Jina AI."""

from sentence_transformers import SentenceTransformer


# Embed text by running inference locally
def embed_locally(
    texts: list[str],
    model_name: str = "jinaai/jina-embeddings-v2-base-en",
) -> list[list[float]]:
    """Run model inference locally.

    Args:
        texts (list[str]): list of texts to be embeddings.
        model_name (str, optional): The embedding model to use.
            Defaults to "jinaai/jina-embeddings-v2-base-en".

    Returns:
        list[list[float]]: embeddings
    """
    model = SentenceTransformer(model_name, trust_remote_code=True)

    model.max_seq_length = 1000  # Chunk Size

    return model.encode(texts).tolist()
