from config.db import db


def test_connection():
    try:
        print("Connected to DB:", db.name)
        collections = db.list_collection_names()
        print("Collections:", collections)
    except Exception as e:
        print("Connection error:", e)


if __name__ == "__main__":
    test_connection()
