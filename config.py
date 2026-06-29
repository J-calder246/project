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
        "mongodb+srv://2511607_db_user:2p9F5hfYzPkRHGMY@cluster0.gqzgwsi.mongodb.net/",
    )
    
    MONGO_DB = "Mortgage_datasets"
    MONGO_STROKE_COLLECTION = "Mortgage_data"
    



MONGO_CLIENT = None