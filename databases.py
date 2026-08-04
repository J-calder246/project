from config import Config
from pymongo import MongoClient

client = MongoClient(Config.MONGO_URI)
mdb = client[Config.MONGO_DB]
input_collection = mdb[Config.INPUT_COLLECTION]