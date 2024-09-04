import datetime
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
        doc: Document = Document(id=str(id), content=json.dumps(mongo_doc, default=str), meta={"name": mongo_doc["name"]})
        return doc

    def get_user_by_email(self, email: str):
        collection = self.get_collection("users")
        user = collection.find_one({"email": email})
        return user

    def save_booking(self, event_id: str, user_id: str, price: float, number_of_tickets: int = 1, sr_citizen_tickets: int = 0, child_tickets: int = 0, student_tickets: int = 0, foreigner_tickets: int = 0):
        collection = self.get_collection("bookings")
        booking = {
            "userId": user_id,
            "eventId": event_id,
            "price": price,
            "numberOfTickets": number_of_tickets,
            "srCitizenTickets": sr_citizen_tickets,
            "childTickets": child_tickets,
            "studentTickets": student_tickets,
            "foreignerTickets": foreigner_tickets,
            "bookingDate": datetime.now(),
            "bookingTime": datetime.now().strftime("%H:%M:%S"),
        }
        collection.insert_one(booking)
        return booking
    
    def get_event_by_id(self, event_id: str):
        collection = self.get_collection("events")
        event = collection.find_one({"_id": event_id})
        return event

