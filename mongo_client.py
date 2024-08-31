import json
from typing import Optional, Any

from bson import ObjectId
from haystack import Document
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


class MongoDBClient:
    def __init__(self, uri: str):
        self.uri = uri
        self.db: Optional[MongoClient] = None

    def connect(self):
        self.db = MongoClient(self.uri, server_api=ServerApi('1'))

    def get_collection(self, collection_name: str):
        return self.db.test[collection_name]

    def close(self):
        self.db.close()

    @staticmethod
    def mongo_event_doc_to_haystack_doc(mongo_doc: dict[str, Any]) -> Document:
        id: ObjectId = mongo_doc.pop("_id")
        doc: Document = Document(id=str(id), content=json.dumps(mongo_doc, default=str))
        # doc.content = json.dumps({"name": name, "description": description})
        return doc
