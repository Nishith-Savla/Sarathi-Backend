# import os
#
# from haystack import Pipeline
# from haystack.components.builders import PromptBuilder
# from haystack.components.converters import HTMLToDocument
# from haystack.components.fetchers import LinkContentFetcher
# from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
# from haystack.components.writers import DocumentWriter
# from haystack.document_stores.in_memory import InMemoryDocumentStore
# from haystack.utils.auth import Secret
# from haystack_integrations.components.generators.google_ai.chat.gemini import \
#     GoogleAIGeminiChatGenerator
#
# from converters.prompt_to_chatmessage_converter import PromptToChatMessage
# from embedders.gemini_document_embedder import GeminiDocumentEmbedder
#
# from embedders.gemini_text_embedder import GeminiTextEmbedder
#
# prompt = """
# Answer the question based on the provided context.
# Context:
# {% for doc in documents %}
#    {{ doc.content }}
# {% endfor %}
# Question: {{ query }}
# """
#
# message_list = []
# document_store = InMemoryDocumentStore()
#
# fetcher = LinkContentFetcher()
# converter = HTMLToDocument()
# embedder = GeminiDocumentEmbedder(api_key=Secret.from_env_var("GOOGLE_API_KEY"))
# writer = DocumentWriter(document_store=document_store)
#
# indexing = Pipeline()
# indexing.add_component("fetcher", fetcher)
# indexing.add_component("converter", converter)
# indexing.add_component("embedder", embedder)
# indexing.add_component("writer", writer)
#
# indexing.connect("fetcher.streams", "converter.sources")
# indexing.connect("converter", "embedder")
# indexing.connect("embedder", "writer")
# indexing.run({
#     "fetcher": {
#         "urls": [
#             "https://haystack.deepset.ai/integrations/cohere",
#             "https://haystack.deepset.ai/integrations/anthropic",
#             "https://haystack.deepset.ai/integrations/jina",
#             "https://haystack.deepset.ai/integrations/nvidia",
#         ]
#     }
# },
# )

from haystack import Pipeline, Document
from haystack.components.builders import PromptBuilder
from haystack.utils.auth import Secret
from haystack_integrations.components.generators.google_ai.chat.gemini import \
    GoogleAIGeminiChatGenerator
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore
from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever

from converters.prompt_to_chatmessage_converter import PromptToChatMessage
from embedders.gemini_text_embedder import GeminiTextEmbedder


class RAGService:
    def __init__(self, env_var_name: str, prompt: str, model: str = "gemini-1.5-flash"):
        self.api_key = Secret.from_env_var(env_var_name)
        self.prompt = prompt
        self.message_list = []
        self.document_store = WeaviateDocumentStore(url="http://127.0.0.1:8080")
        self.model = model

        self.query_embedder = GeminiTextEmbedder(api_key=self.api_key)
        self.retriever = WeaviateEmbeddingRetriever(document_store=self.document_store)
        self.prompt_builder = PromptBuilder(template=self.prompt)
        self.prompt_to_chat_message_converter = PromptToChatMessage(prompt=self.prompt,
                                                                    message_list=self.message_list)
        self.generator = GoogleAIGeminiChatGenerator(model=self.model)

        self.rag = Pipeline()
        self.rag.add_component("query_embedder", self.query_embedder)
        self.rag.add_component("retriever", self.retriever)
        self.rag.add_component("prompt", self.prompt_builder)
        self.rag.add_component("prompt_to_chat_message_converter",
                               self.prompt_to_chat_message_converter)
        self.rag.add_component("generator", self.generator)

        self.rag.connect("query_embedder.embedding", "retriever.query_embedding")
        self.rag.connect("retriever.documents", "prompt.documents")
        self.rag.connect("prompt.prompt", "prompt_to_chat_message_converter")
        self.rag.connect("prompt_to_chat_message_converter.chat_messages", "generator")

    def query(self, question: str):
        result = self.rag.run({
            "query_embedder": {"text": question},
            "retriever": {"top_k": 1},
            "prompt": {"query": question},
        })
        return result["generator"]["replies"][0]

    def add_documents(self, documents: _DocumentType):
        self.document_store.write_documents(documents)
