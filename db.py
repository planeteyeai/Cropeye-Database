import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise Exception("DATABASE_URL missing")

conn = psycopg2.connect(DATABASE_URL)

cursor = conn.cursor()

cursor.execute("SELECT * FROM plots LIMIT 5")

rows = cursor.fetchall()

for row in rows:
    print(row)

cursor.close()
conn.close()
