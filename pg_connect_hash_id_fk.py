import json
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
import os

load_dotenv("db_localhost.env")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", 5432))
}

JSON_FILE = r"C:\ai_project\hash_id_event_articles.json"

try:

    conn = psycopg2.connect(**DB_CONFIG)

    cursor = conn.cursor()

    print("Connected to PostgreSQL")

    cursor.execute("""

        DROP TABLE IF EXISTS hash_id_foreign_key;

        CREATE TABLE hash_id_foreign_key (

            hash_id VARCHAR(150) PRIMARY KEY,

            article_url TEXT,

            headline_english TEXT,

            article_summary TEXT,

            event_id VARCHAR(20) NOT NULL

        );

    """)

    conn.commit()

    print("Table created successfully")

    with open(JSON_FILE, "r", encoding="utf-8") as f:

        data = json.load(f)

    inserted = 0
    skipped = 0
    failed = 0

    for item in data:

        try:

            cursor.execute("""

                INSERT INTO hash_id_foreign_key (

                    hash_id,
                    article_url,
                    headline_english,
                    article_summary,
                    event_id

                )

                VALUES (

                    %s,
                    %s,
                    %s,
                    %s,
                    %s

                )

                ON CONFLICT (hash_id)
                DO NOTHING;

            """, (

                item.get("hash_id"),
                item.get("article_url"),
                item.get("headline_english"),
                item.get("article_summary"),
                item.get("event_id")

            ))

            if cursor.rowcount == 0:

                skipped += 1

            else:

                inserted += 1

        except Exception as e:

            failed += 1

            print(
                f"Insert Error ({item.get('hash_id')}):",
                e
            )

    conn.commit()

    print("\nFINAL REPORT")

    print("Inserted :", inserted)
    print("Skipped  :", skipped)
    print("Failed   :", failed)

    cursor.execute("""

        SELECT

            hash_id,
            headline_english,
            event_id

        FROM hash_id_foreign_key

        ORDER BY event_id

        LIMIT 20;

    """)

    rows = cursor.fetchall()

    print("\nSTORED DATA\n")

    for row in rows:

        print(row)

except Error as e:

    print("POSTGRESQL ERROR:", e)

except Exception as e:

    print("GENERAL ERROR:", e)

finally:

    try:

        if conn:

            cursor.close()
            conn.close()

            print("\nPostgreSQL connection closed")

    except:
        pass