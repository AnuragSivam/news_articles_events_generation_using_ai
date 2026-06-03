import pandas as pd
import psycopg2
import os
import re
import spacy
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# LOAD ENV
# =========================

load_dotenv("db_localhost.env")

# =========================
# DB CONNECTION
# =========================

conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    database=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT")
)

cursor = conn.cursor()

# =========================
# DROP OLD TABLE IF EXISTS
# =========================

cursor.execute("""
DROP TABLE IF EXISTS articles_events_tags_all;
""")

conn.commit()

# =========================
# CREATE NEW TABLE
# =========================

cursor.execute("""
CREATE TABLE articles_events_tags_all (

    event_id VARCHAR(20) PRIMARY KEY,

    event_tag TEXT,

    event_type VARCHAR(100),

    article_nos TEXT,

    hash_ids TEXT,

    published_dates TEXT,

    published_times TEXT,

    places TEXT,

    event_summary TEXT

);
""")

conn.commit()

print("\nTable 'articles_events_tags_all' created successfully.")

# =========================
# LOAD NLP + EMBEDDING MODEL
# =========================

print("\nLoading NLP model...")

nlp = spacy.load("en_core_web_sm")

print("Loading Sentence Transformer model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# =========================
# FETCH ARTICLES
# =========================

query = """
SELECT
    article_no,
    hash_id,
    headline_english,
    body_content_english,
    keywords,
    subject_text,
    verb_text,
    object_text,
    published_date,
    published_time
FROM mzamin_articles_english
ORDER BY article_no::INTEGER ASC
"""

df = pd.read_sql(query, conn)

print(f"\nTotal Articles Found: {len(df)}")

# =========================
# HANDLE NULL VALUES
# =========================

columns_to_clean = [

    "headline_english",
    "body_content_english",
    "keywords",
    "subject_text",
    "verb_text",
    "object_text"

]

for col in columns_to_clean:

    df[col] = df[col].fillna("")

# =========================
# COMBINE TEXT
# =========================

df["combined_text"] = (

    df["headline_english"] + " " +

    df["keywords"] + " " +

    df["subject_text"] + " " +

    df["verb_text"] + " " +

    df["object_text"]

)

texts = df["combined_text"].tolist()

# =========================
# GENERATE EMBEDDINGS
# =========================

print("\nGenerating embeddings...")

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

print("\nCalculating similarity matrix...")

similarity_matrix = cosine_similarity(
    embeddings
)

# =========================
# EVENT TAG GENERATOR
# =========================

def generate_event_tag(text):

    if not text:
        return "GENERAL_EVENT"

    text = text.lower()

    words = re.findall(
        r'\b[a-z]+\b',
        text
    )

    stop_words = {

        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "have",
        "were",
        "been",
        "into",
        "about",
        "their",
        "there",
        "would",
        "could",
        "should",
        "after",
        "before",
        "said",
        "says",
        "will",
        "also",
        "more",
        "than",
        "news",
        "today",
        "update"

    }

    important_words = []

    for word in words:

        if (
            len(word) > 3 and
            word not in stop_words
        ):

            important_words.append(
                word.upper()
            )

    unique_words = []

    for word in important_words:

        if word not in unique_words:

            unique_words.append(word)

    tag_words = unique_words[:3]

    if len(tag_words) == 0:

        return "GENERAL_EVENT"

    return "_".join(tag_words)

# =========================
# IMPROVED PLACE EXTRACTION
# =========================

def extract_places(text):

    if not text or pd.isna(text):

        return "UNKNOWN"

    doc = nlp(str(text))

    places = set()

    for ent in doc.ents:

        if ent.label_ in [

            "GPE",
            "LOC",
            "FAC"

        ]:

            cleaned_place = ent.text.strip()

            if len(cleaned_place) > 1:

                places.add(cleaned_place)

    fallback_places = [

        "Dhaka",
        "Bangladesh",
        "India",
        "Pakistan",
        "China",
        "USA",
        "London",
        "Delhi",
        "Assam",
        "Gaza",
        "Israel",
        "Russia",
        "Ukraine",
        "Myanmar",
        "Nepal"

    ]

    text_lower = text.lower()

    for place in fallback_places:

        if place.lower() in text_lower:

            places.add(place)

    if len(places) == 0:

        return "UNKNOWN"

    return " | ".join(
        sorted(places)
    )

# =========================
# EVENT SUMMARY GENERATOR
# =========================

def generate_event_summary(cluster_indices):

    summary_parts = []

    for idx in cluster_indices:

        headline = str(
            df.iloc[idx]["headline_english"]
        ).strip()

        body = str(
            df.iloc[idx]["body_content_english"]
        ).strip()

        # Take first 500 chars
        short_body = body[:500]

        combined = f"{headline}. {short_body}"

        summary_parts.append(combined)

    final_summary = "\n\n".join(summary_parts)

    # Limit summary size
    if len(final_summary) > 4000:

        final_summary = final_summary[:4000]

    return final_summary

# =========================
# EVENT CLUSTERING
# =========================

visited = set()

event_counter = 1

SIMILARITY_THRESHOLD = 0.70

print("\nStarting event clustering...\n")

for i in range(len(df)):

    if i in visited:
        continue

    cluster_articles = []
    cluster_hashes = []
    cluster_dates = []
    cluster_times = []
    cluster_places = []
    cluster_indices = []

    current_text = str(
        df.iloc[i]["combined_text"]
    )

    visited.add(i)

    cluster_indices.append(i)

    # =========================
    # ADD CURRENT ARTICLE
    # =========================

    cluster_articles.append(
        str(df.iloc[i]["article_no"])
    )

    cluster_hashes.append(
        str(df.iloc[i]["hash_id"])
    )

    cluster_dates.append(
        str(df.iloc[i]["published_date"])
    )

    cluster_times.append(
        str(df.iloc[i]["published_time"])
    )

    cluster_places.append(
        extract_places(current_text)
    )

    # =========================
    # FIND SIMILAR ARTICLES
    # =========================

    for j in range(i + 1, len(df)):

        if j in visited:
            continue

        similarity_score = similarity_matrix[i][j]

        if similarity_score >= SIMILARITY_THRESHOLD:

            visited.add(j)

            cluster_indices.append(j)

            cluster_articles.append(
                str(df.iloc[j]["article_no"])
            )

            cluster_hashes.append(
                str(df.iloc[j]["hash_id"])
            )

            cluster_dates.append(
                str(df.iloc[j]["published_date"])
            )

            cluster_times.append(
                str(df.iloc[j]["published_time"])
            )

            cluster_places.append(
                extract_places(
                    str(
                        df.iloc[j]["combined_text"]
                    )
                )
            )

    # =========================
    # CLEAN PLACES
    # =========================

    final_places = sorted(set([

        p.strip()

        for place_group in cluster_places

        for p in str(place_group).split("|")

        if p.strip() and p.strip() != "UNKNOWN"

    ]))

    if len(final_places) == 0:

        final_places = ["UNKNOWN"]

    # =========================
    # EVENT TYPE
    # =========================

    if len(cluster_articles) > 1:

        event_type = "RELATED_ARTICLES_EVENT"

    else:

        event_type = "SINGLE_ARTICLE_EVENT"

    # =========================
    # EVENT ID
    # =========================

    event_id = f"EV{event_counter:05d}"

    event_counter += 1

    # =========================
    # EVENT TAG
    # =========================

    event_tag = generate_event_tag(
        current_text
    )

    # =========================
    # EVENT SUMMARY
    # =========================

    event_summary = generate_event_summary(
        cluster_indices
    )

    # =========================
    # INSERT INTO DATABASE
    # =========================

    cursor.execute(
        """
        INSERT INTO articles_events_tags_all (

            event_id,
            event_tag,
            event_type,
            article_nos,
            hash_ids,
            published_dates,
            published_times,
            places,
            event_summary

        )

        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            event_id,

            event_tag,

            event_type,

            " | ".join(cluster_articles),

            " | ".join(cluster_hashes),

            " | ".join(cluster_dates),

            " | ".join(cluster_times),

            " | ".join(final_places),

            event_summary
        )
    )

    conn.commit()

    print(
        f"{event_id} -> "
        f"{event_tag} -> "
        f"{event_type} -> "
        f"{len(cluster_articles)} article(s)"
    )

# =========================
# CLOSE CONNECTION
# =========================

cursor.close()

conn.close()

print("\nEVENT TAGGING + SUMMARIZATION COMPLETED SUCCESSFULLY")