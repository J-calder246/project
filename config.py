import os
# Config class to hold application

class Config:
    BASE_DIR = os.getcwd()
    SECRET_KEY = os.environ.get("SECRET_KEY", "Buglady458")

    DB_PATH = os.path.join(BASE_DIR, "users.db")
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    ALLOWED_EXTENSIONS = {"csv"}
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Mongo configuration (URI only does not connect here)
    MONGO_URI = os.environ.get(
        "MONGO_URI",
        "mongodb+srv://2511607_db_user:Izp9wYNoNgybKG6d@cluster0.ekuobcv.mongodb.net/"
    )
    
    MONGO_DB = "mortgageDB"
    INPUT_COLLECTION = "inputcollection"
    INPUT_COLLECTION_ENCODED = "inputcollection(with dummies)"
    



MONGO_CLIENT = None

