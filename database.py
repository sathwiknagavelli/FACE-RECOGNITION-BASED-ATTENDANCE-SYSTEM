import mysql.connector

DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "",
    "database": "face_recognition_db",
    "port": 3306
}


def connect_db():
    return mysql.connector.connect(**DB_CONFIG)


def setup_database():
    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        port=DB_CONFIG["port"]
    )
    cursor = conn.cursor()

    cursor.execute("CREATE DATABASE IF NOT EXISTS face_recognition_db")
    cursor.execute("USE face_recognition_db")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            email VARCHAR(100) NOT NULL,
            fullname VARCHAR(100) NOT NULL,
            username VARCHAR(100) NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    """)

    conn.commit()
    cursor.close()
    conn.close()