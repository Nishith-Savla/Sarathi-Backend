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
import json
from typing import Any

from haystack import Pipeline, Document
from haystack.components.builders import PromptBuilder
from haystack.components.joiners import BranchJoiner
from haystack.components.validators import JsonSchemaValidator
from haystack.dataclasses import ChatMessage
from haystack.document_stores.types import DuplicatePolicy
from haystack.utils.auth import Secret
from haystack_integrations.components.generators.google_ai.chat.gemini import \
    GoogleAIGeminiChatGenerator
from haystack_integrations.components.retrievers.weaviate import WeaviateEmbeddingRetriever
from haystack_integrations.document_stores.weaviate import WeaviateDocumentStore

from converters.prompt_to_chatmessage_converter import PromptToChatMessage
from embedders.gemini_document_embedder import GeminiDocumentEmbedder
from embedders.gemini_text_embedder import GeminiTextEmbedder
from mongo_client import MongoDBClient


class RAGService:
    def __init__(self, env_var_name: str, prompt: str, system_prompt: str = None, output_schema: dict[str, Any] = None,
                 model: str = "gemini-1.5-pro", generation_config: dict[str, Any] = None):
        self.api_key = Secret.from_env_var(env_var_name)
        self.prompt = prompt
        self.document_store = WeaviateDocumentStore(url="http://127.0.0.1:8080")
        self.model = model
        self.system_prompt = system_prompt
        self.output_schema = output_schema

        self.document_embedder = GeminiDocumentEmbedder(api_key=self.api_key)
        self.query_embedder = GeminiTextEmbedder(api_key=self.api_key)
        self.retriever = WeaviateEmbeddingRetriever(document_store=self.document_store)
        self.prompt_builder = PromptBuilder(template=self.prompt)
        self.prompt_to_chat_message_converter = PromptToChatMessage(prompt=self.prompt)
        self.generator = GoogleAIGeminiChatGenerator(model=self.model, generation_config=generation_config)
        self.branch_joiner = BranchJoiner(list[ChatMessage])
        self.schema_validator = JsonSchemaValidator()

        self.pipeline = Pipeline()
        # self.rag.add_component("query_embedder", self.query_embedder)
        # self.rag.add_component("retriever", self.retriever)
        self.pipeline.add_component("prompt", self.prompt_builder)
        self.pipeline.add_component("prompt_to_chat_message_converter",
                                    self.prompt_to_chat_message_converter)
        self.pipeline.add_component("generator", self.generator)
        self.pipeline.add_component("schema_validator", self.schema_validator)
        self.pipeline.add_component("branch_joiner", self.branch_joiner)

        # self.rag.connect("query_embedder.embedding", "retriever.query_embedding")
        # self.rag.connect("retriever.documents", "prompt.documents")
        # self.rag.connect("prompt.prompt", "prompt_to_chat_message_converter")
        # self.rag.connect("prompt_to_chat_message_converter.chat_messages", "generator")

        self.pipeline.connect("prompt.prompt", "prompt_to_chat_message_converter")
        self.pipeline.connect("prompt_to_chat_message_converter.message_list", "branch_joiner")
        self.pipeline.connect("branch_joiner", "generator")
        self.pipeline.connect("generator.replies", "schema_validator.messages")
        self.pipeline.connect("schema_validator.validation_error", "branch_joiner")

    def __del__(self):
        try:
            self.document_store.client.close()
        except Exception:
            pass
    
    def _parse_output(self, result):
        validated_message: ChatMessage = result["schema_validator"]["validated"][0]
        validated_message.content = validated_message.content.strip()
        message_list = result["prompt_to_chat_message_converter"]["message_list"]
        message_list.append(validated_message)
        return message_list

    def new_chat(self):
        result = self.pipeline.run({
            "prompt": {"template": self.system_prompt,
                       "template_variables": {"documents": self.view_documents()}},
            "prompt_to_chat_message_converter": {"message_list": [], "role": "system"},
            "schema_validator": self.output_schema,
        }, include_outputs_from={"prompt_to_chat_message_converter", "generator"})
        return self._parse_output(result)

    def query(self, question: str, message_list: list[dict[str, str]]):
        message_list = [ChatMessage.from_dict(message) for message in message_list]
        result = self.pipeline.run({
            "prompt": {"query": question},
            "prompt_to_chat_message_converter": {"message_list": message_list, "role": "user"},
            "schema_validator": self.output_schema,
        }, include_outputs_from={"prompt_to_chat_message_converter", "generator"})
        return self._parse_output(result)

    def view_documents(self, filters: dict[str, Any] | None = None):
        return self.document_store.filter_documents(filters=filters)

    def add_documents(self, documents: list[Document]):
        self.document_store.write_documents(documents, policy=DuplicatePolicy.OVERWRITE)

    def delete_documents(self, document_ids: list[str]):
        self.document_store.delete_documents(document_ids=document_ids)

    def refresh_document_store(self, mongo_client: MongoDBClient):
        events = mongo_client.get_collection("events")
        docs = \
            self.document_embedder.run(documents=[mongo_client.mongo_event_doc_to_haystack_doc(doc)
                                                  for doc in events.find()])["documents"]
        self.add_documents(docs)
