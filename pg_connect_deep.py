import pandas as pd
import psycopg2
from psycopg2 import Error
from dotenv import load_dotenv
import os

# =========================
# LOAD ENV
# =========================

load_dotenv("db_localhost.env")

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "dbname": os.getenv("DB_NAME"),
    "port": int(os.getenv("DB_PORT", 5432))
}

# =========================
# CSV FILE
# =========================

CSV_FILE = r"C:\ai_project\articles_nlp_output.csv"

# =========================
# CONNECT DATABASE
# =========================

try:

    conn = psycopg2.connect(**DB_CONFIG)

    cursor = conn.cursor()

    print("Connected to PostgreSQL")

    # =========================
    # CREATE TABLE
    # =========================

    cursor.execute("""

        DROP TABLE IF EXISTS mzamin_articles_nlp;

        CREATE TABLE IF NOT EXISTS mzamin_articles_nlp (

            article_no VARCHAR(20) PRIMARY KEY,

            hash_id VARCHAR(150),

            article_url TEXT,

            headline TEXT,

            headline_english TEXT,

            body_content TEXT,

            body_content_english TEXT,

            published_date DATE,

            published_time TIME,

            article_summary TEXT,

            keywords TEXT,

            pos TEXT,

            reporter_name TEXT,

            image_url TEXT,

            image_caption TEXT

        );

    """)

    conn.commit()

    print("Table created successfully")

    # =========================
    # LOAD CSV
    # =========================

    df = pd.read_csv(CSV_FILE)

    inserted = 0
    failed = 0
    skipped = 0

    # =========================
    # INSERT DATA
    # =========================

    for _, row in df.iterrows():

        try:

            cursor.execute("""

                INSERT INTO mzamin_articles_nlp (

                    article_no,
                    hash_id,
                    article_url,
                    headline,
                    headline_english,
                    body_content,
                    body_content_english,
                    published_date,
                    published_time,
                    article_summary,
                    keywords,
                    pos,
                    reporter_name,
                    image_url,
                    image_caption

                )

                VALUES (

                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s

                )

                ON CONFLICT (article_no)
                DO NOTHING;

            """, (

                row.get("article_no"),
                row.get("hash_id"),
                row.get("article_url"),
                row.get("headline"),
                row.get("headline_english"),
                row.get("body_content"),
                row.get("body_content_english"),
                row.get("published_date"),
                row.get("published_time"),
                row.get("article_summary"),
                row.get("keywords"),
                row.get("POS"),
                row.get("reporter_name"),
                row.get("image_url"),
                row.get("image_caption")

            ))

            if cursor.rowcount == 0:

                skipped += 1

                print(f"Skipped: {row.get('article_no')}")

            else:

                inserted += 1

                print(f"Inserted: {row.get('article_no')}")

        except Exception as e:

            failed += 1

            print(f"Insert Error ({row.get('article_no')}):", e)

            conn.rollback()

    conn.commit()

    # =========================
    # FINAL REPORT
    # =========================

    print("\nFINAL REPORT")

    print("Inserted :", inserted)
    print("Skipped  :", skipped)
    print("Failed   :", failed)

    # =========================
    # VERIFY DATA
    # =========================

    cursor.execute("""

        SELECT

            article_no,
            headline_english,
            article_summary,
            keywords,
            pos

        FROM mzamin_articles_nlp

        ORDER BY article_no

        LIMIT 10;

    """)

    rows = cursor.fetchall()

    print("\nSTORED DATA\n")

    for row in rows:

        print(row)

# =========================
# ERRORS
# =========================

except Error as e:

    print("POSTGRESQL ERROR:", e)

except Exception as e:

    print("GENERAL ERROR:", e)

# =========================
# CLOSE CONNECTION
# =========================

finally:

    try:

        if conn:

            cursor.close()

            conn.close()

            print("\nPostgreSQL connection closed")

    except:
        pass