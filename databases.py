"""
Section that sefines to routes to the MongoDB collections set up in config
"""

from config import Config
from pymongo import MongoClient

#defining routes into the MongoDB collections
_client = MongoClient(Config.MONGO_URI)
_mdb = _client[Config.MONGO_DB]
input_collection = _mdb[Config.INPUT_COLLECTION]
input_collection_encoded = _mdb[Config.INPUT_COLLECTION_ENCODED]

"""
#testing
_client.admin.command("ping")
print("MongoDB connection successful")

result = input_collection.insert_one({"test": "working", "number": 1})
print("Inserted ID:", result.inserted_id)

"""