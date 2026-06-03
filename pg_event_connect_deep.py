import pandas as pd
import psycopg2

from psycopg2 import Error

from dotenv import load_dotenv

import os

# =========================================================
# LOAD ENV
# =========================================================

load_dotenv("db_localhost.env")

DB_CONFIG = {

    "host": os.getenv("DB_HOST"),

    "user": os.getenv("DB_USER"),

    "password": os.getenv("DB_PASSWORD"),

    "dbname": os.getenv("DB_NAME"),

    "port": int(
        os.getenv("DB_PORT", 5432)
    )

}

# =========================================================
# CSV FILE
# =========================================================

CSV_FILE = r"C:\ai_project\articles_events_tags_all_processed_new_deep1.csv"

# =========================================================
# CONNECT DATABASE
# =========================================================

try:

    conn = psycopg2.connect(
        **DB_CONFIG
    )

    cursor = conn.cursor()

    print(
        "\nConnected to PostgreSQL"
    )

    # =====================================================
    # CREATE TABLE
    # =====================================================

    cursor.execute("""

        DROP TABLE IF EXISTS mzamin_articles_events_deep;

        CREATE TABLE IF NOT EXISTS mzamin_articles_events_deep (

            id SERIAL PRIMARY KEY,

            event_id VARCHAR(20),

            event_tag TEXT,

          

            article_nos TEXT,

            hash_ids TEXT,

            headline_english TEXT,

            published_dates TEXT,

            published_times TEXT,

            common_places TEXT,

            event_summary TEXT,

            
            total_articles INTEGER

        );

    """)

    conn.commit()

    print(
        "\nTable created successfully"
    )

    # =====================================================
    # LOAD CSV
    # =====================================================

    df = pd.read_csv(
        CSV_FILE
    )

    inserted = 0

    failed = 0

    # =====================================================
    # INSERT DATA
    # =====================================================

    for _, row in df.iterrows():

        try:

            cursor.execute("""

                INSERT INTO mzamin_articles_events_deep (

                    event_id,
                    event_tag,
                    event_type,
                    article_nos,
                    hash_ids,
                    headline_english,
                    published_dates,
                    published_times,
                    common_places,
                    event_summary,
                    similarity_score,
                    total_articles

                )

                VALUES (

                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s

                );

            """, (

                row.get("event_id"),

                row.get("event_tag"),

                row.get("event_type"),

                row.get("article_nos"),

                row.get("hash_ids"),

                row.get("headline_english"),

                row.get("published_dates"),

                row.get("published_times"),

                row.get("common_places"),

                row.get("event_summary"),

                float(
                    row.get(
                        "similarity_score",
                        0
                    )
                ),

                int(
                    row.get(
                        "total_articles",
                        1
                    )
                )

            ))

            inserted += 1

            print(
                f"Inserted: "
                f"{row.get('event_id')}"
            )

        except Exception as e:

            failed += 1

            print(

                f"\nInsert Error "
                f"({row.get('event_id')}):"

            )

            print(e)

            conn.rollback()

    conn.commit()

    # =====================================================
    # FINAL REPORT
    # =====================================================

    print("\nFINAL REPORT")

    print(
        f"Inserted : {inserted}"
    )

    print(
        f"Failed   : {failed}"
    )

    # =====================================================
    # VERIFY DATA
    # =====================================================

    cursor.execute("""

        SELECT

            event_id,
            event_tag,
            event_type,
            similarity_score,
            total_articles

        FROM mzamin_articles_events_deep

        ORDER BY id

        LIMIT 10;

    """)

    rows = cursor.fetchall()

    print("\nSTORED DATA\n")

    for row in rows:

        print(row)

# =========================================================
# ERRORS
# =========================================================

except Error as e:

    print(
        "\nPOSTGRESQL ERROR:"
    )

    print(e)

except Exception as e:

    print(
        "\nGENERAL ERROR:"
    )

    print(e)

# =========================================================
# CLOSE CONNECTION
# =========================================================

finally:

    try:

        if conn:

            cursor.close()

            conn.close()

            print(
                "\nPostgreSQL connection closed"
            )

    except:
        pass