from typing import Any, Dict, List, Optional

import google.generativeai as genai
from haystack import component, default_from_dict, default_to_dict
from haystack.utils import Secret
from tqdm import tqdm


@component
class GeminiTextEmbedder:
    def __init__(
            self,
            api_key: Secret = Secret.from_env_var("GOOGLE_API_KEY"),
            model: str = "models/text-embedding-004",
            prefix: str = "",
            suffix: str = "",
            batch_size: int = 32,
            progress_bar: bool = True,
    ):
        """
        Initialize the GeminiTextEmbedder component.

        :param api_key:
            The Gemini (GOOGLE) API key.
            You can set it with an environment variable `GOOGLE_API_KEY`, or pass with this parameter
            during initialization.
        :param model:
            The name of the model to use for calculating embeddings.
            The default model is `models/text-embedding-004`.
        :param prefix:
            A string to add at the beginning of each text to embed.
        :param suffix:
            A string to add at the end of each text to embed.
        :param batch_size:
            Number of texts to process at once.
        :param progress_bar:
            If `True` shows a progress bar when running.
        """
        self.api_key = api_key
        self.model = model
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar

        genai.configure(api_key=api_key.resolve_value())

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(self, api_key=self.api_key, model=self.model)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeminiTextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, texts: List[str]) -> List[str]:
        """
        Prepare the texts to embed by adding prefixes and suffixes.
        """
        texts_to_embed = [
            (self.prefix + text + self.suffix).replace("\n", " ") for text in texts
        ]
        return texts_to_embed

    def _embed_batch(self, texts_to_embed: List[str], batch_size: int) -> List[List[float]]:
        """
        Embed a list of texts in batches.
        """
        all_embeddings = []
        for i in tqdm(
                range(0, len(texts_to_embed), batch_size), disable=not self.progress_bar,
                desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i: i + batch_size]
            embeddings = genai.embed_content(content=batch, model=self.model)
            all_embeddings.extend(embeddings['embedding'])

        return all_embeddings

    @component.output_types(embedding=List[float])
    def run(self, text: str) -> Dict[str, Any]:
        """
        Embed a single string.

        :param text: Text to embed.
        :returns: A dictionary with the embedding.
        """
        if not isinstance(text, str):
            raise TypeError(
                "GeminiTextEmbedder expects a string as input."
                "In case you want to embed a list of Documents, please use the GeminiDocumentEmbedder."
            )

        texts_to_embed = self._prepare_texts_to_embed([text])
        embeddings = self._embed_batch(texts_to_embed=texts_to_embed, batch_size=self.batch_size)

        return {"embedding": embeddings[0]}