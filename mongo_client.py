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
        name: str = mongo_doc.pop("name")
        description: str = mongo_doc.pop("description")
        doc: Document = Document.from_dict(mongo_doc)
        doc.id = id
        doc.content = json.dumps({"name": name, "description": description})
        return doc


if __name__ == "__main__":
    uri = "mongodb+srv://nbokdeb21:sharayu2000@cluster0.zpsol.mongodb.net/test?retryWrites=true&w=majority&appName=Cluster0"
    mongo_client = MongoDBClient(uri=uri)
    mongo_client.connect()
    collection = mongo_client.get_collection(collection_name="events")
    # collection.insert_one({"name": "John"})
    doc = next(collection.find())
    print(mongo_client.mongo_event_doc_to_haystack_doc(doc))
    mongo_client.close()
