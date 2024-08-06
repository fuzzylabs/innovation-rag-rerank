"""Embedding using Jina AI."""

# import json

# import requests
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

    return model.encode(texts)


# # Embed text by using Jina AI's API
# def embed_remotely(
#     texts: list[str],
#     api_key: str,
#     model_name: str = "jina-embeddings-v2-base-en",
# ) -> list[list[float]]:
#     """Run model inference using JanaAI's managed service API.

#     Args:
#         texts (list[str]): list of texts to be embeddings.
#         api_key (str): JanaAI API key.
#         model_name (str, optional): The embedding model to use.
#             Defaults to "jinaai/jina-embeddings-v2-small-en".

#     Returns:
#         list[list[float]]: embeddings
#     """
#     url = "https://api.jina.ai/v1/embeddings"
# headers = {
#     "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"
# }
#     data = {"model": model_name, "embedding_type": "float", "input": texts}

#     response = requests.post(url, headers=headers, data=json.dumps(data))

#     return [data["embedding"] for data in response.json()["data"]]
