from typing import Any, Dict, List, Optional

import google.generativeai as genai
from haystack import component, default_from_dict, default_to_dict, Document
from haystack.utils import Secret
from tqdm import tqdm


@component
class GeminiDocumentEmbedder:
    def __init__(
            self,
            api_key: Secret = Secret.from_env_var("GOOGLE_API_KEY"),
            model: str = "models/text-embedding-004",
            prefix: str = "",
            suffix: str = "",
            batch_size: int = 32,
            progress_bar: bool = True,
            meta_fields_to_embed: Optional[List[str]] = None,
            embedding_separator: str = "\n",
    ):
        """
        Initialize the GeminiEmbedder component.

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
            Number of Documents to process at once.
        :param progress_bar:
            If `True` shows a progress bar when running.
        :param meta_fields_to_embed:
            List of meta fields that will be embedded along with the Document text.
        :param embedding_separator:
            Separator used to concatenate the meta fields to the Document text.
        """
        self.api_key = api_key
        self.model = model
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator

        genai.configure(api_key=api_key.resolve_value())

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(self, api_key=self.api_key, model=self.model)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeminiDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if
                key in doc.meta and doc.meta[key] is not None
            ]

            text_to_embed = (
                    self.prefix + self.embedding_separator.join(
                meta_values_to_embed + [doc.content or ""]) + self.suffix
            )

            text_to_embed = text_to_embed.replace("\n", " ")
            texts_to_embed.append(text_to_embed)
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
            # embed using genai.embed_content
            embeddings = genai.embed_content(content=batch, model=self.model)
            all_embeddings.extend(embeddings['embedding'])

        return all_embeddings

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Embed a list of Documents.

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with embeddings
            - `meta`: Information about the usage of the model.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "GeminiDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a string, please use the GeminiTextEmbedder."
            )

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        embeddings = self._embed_batch(texts_to_embed=texts_to_embed, batch_size=self.batch_size)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents}
