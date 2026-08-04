#File for saving user data from inputs into application

#Connection string
#--------------------
#mongodb+srv://2511607_db_user:eXuTgVM5v5NqJhGv@inputdata.esobqrq.mongodb.net/

import os

class Config:
    BASE_DIRECTORY = os.getcwd()
    SECRET_KEY = os.environ.get("SECRET_KEY", "secret_key_config")
    DB_PATH = os.path.join(BASE_DIRECTORY, "user_data.db")
    UPLOAD_FOLDER = os.path.join(BASE_DIRECTORY, "uploads")
    ALLOWED_EXTENSIONS = {"csv"}
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    #Mongo connection
    MONGO_URI = os.environ.get(
        "MONGO_URI",
        "mongodb+srv://2511607_db_user:eXuTgVM5v5NqJhGv@inputdata.esobqrq.mongodb.net/"

    )

    MONGO_DB = "input_data"
    INPUT_COLLECTION = "input_collection"

MONGO_CLIENT = None
