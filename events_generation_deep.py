import pandas as pd
import numpy as np
import re
import spacy

from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

INPUT_CSV = r"C:\ai_project\mzamin_article_nlp_pg.csv"

OUTPUT_CSV = r"C:\ai_project\articles_events_tags_all_processed_new_deep.csv"

SIMILARITY_THRESHOLD = 0.85

MIN_CLUSTER_SIZE = 2

MAX_CLUSTER_SIZE = 8

print("\nLoading CSV...\n")

df = pd.read_csv(INPUT_CSV)

df = df.fillna("")

ORIGINAL_TOTAL_ARTICLES = len(df)

print(f"\nTOTAL INPUT ARTICLES: {ORIGINAL_TOTAL_ARTICLES}")

print("\nLoading spaCy...\n")

nlp = spacy.load("en_core_web_sm")

print("\nLoading BGE model...\n")

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5"
)

def clean_text(text):

    text = str(text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def create_event_text(row):

    headline = clean_text(
        row.get("headline_english", "")
    )

    summary = clean_text(
        row.get("article_summary", "")
    )

    body = clean_text(
        row.get("body_content", "")
    )

    weighted_headline = (
        (headline + " ") * 4
    )

    combined = f"""
    {weighted_headline}

    SUMMARY:
    {summary}

    BODY:
    {body[:3000]}
    """

    return combined.strip()

df["event_text"] = df.apply(
    create_event_text,
    axis=1
)

def generate_event_tag(cluster_df):

    headline = str(
        cluster_df.iloc[0]["headline_english"]
    ).strip()

    headline = clean_text(headline)

    headline_lower = headline.lower()

    doc = nlp(headline)

    places = []
    orgs = []
    persons = []
    nouns = []

    for ent in doc.ents:

        text = ent.text.upper()

        text = re.sub(
            r"[^A-Z0-9 ]",
            "",
            text
        ).strip()

        text = text.replace(" ", "_")

        if len(text) < 3:
            continue

        if ent.label_ in ["GPE", "LOC"]:

            places.append(text)

        elif ent.label_ in ["ORG", "NORP"]:

            orgs.append(text)

        elif ent.label_ == "PERSON":

            persons.append(text)

    for token in doc:

        if token.pos_ == "NOUN":

            word = token.lemma_.upper()

            word = re.sub(
                r"[^A-Z0-9]",
                "",
                word
            )

            if len(word) > 3:

                nouns.append(word)

    EVENT_PATTERNS = {

        "protest": "PROTEST",
        "rally": "RALLY",
        "election": "ELECTION",
        "vote": "ELECTION",
        "murder": "MURDER",
        "kill": "KILLING",
        "death": "DEATH",
        "dead": "DEATH",
        "crash": "CRASH",
        "accident": "ACCIDENT",
        "collision": "ACCIDENT",
        "fire": "FIRE",
        "blast": "BLAST",
        "explosion": "EXPLOSION",
        "attack": "ATTACK",
        "airstrike": "AIRSTRIKE",
        "bomb": "BOMBING",
        "arrest": "ARREST",
        "resign": "RESIGNATION",
        "flood": "FLOOD",
        "cyclone": "CYCLONE",
        "earthquake": "EARTHQUAKE",
        "storm": "STORM",
        "match": "MATCH",
        "game": "MATCH",
        "football": "FOOTBALL_MATCH",
        "cricket": "CRICKET_MATCH",
        "world cup": "WORLD_CUP",
        "ssc": "SSC_EXAM",
        "hsc": "HSC_EXAM",
        "exam": "EXAM",
        "measles": "MEASLES_OUTBREAK",
        "dengue": "DENGUE_OUTBREAK",
        "covid": "COVID_OUTBREAK",
        "shooting": "SHOOTING",
        "gas": "GAS_EXPLOSION"
    }

    detected_event = None

    for key, value in EVENT_PATTERNS.items():

        if key in headline_lower:

            detected_event = value

            break

    prefix = None

    if len(places) > 0:

        prefix = places[0]

    elif len(orgs) > 0:

        prefix = orgs[0]

    elif len(persons) > 0:

        prefix = persons[0]

    elif len(nouns) > 0:

        prefix = nouns[0]

    if prefix and detected_event:

        final_tag = f"{prefix}_{detected_event}"

    elif detected_event and len(nouns) > 0:

        final_tag = f"{nouns[0]}_{detected_event}"

    elif prefix and len(nouns) > 1:

        final_tag = f"{prefix}_{nouns[1]}"

    else:

        chunks = []

        for chunk in doc.noun_chunks:

            text = chunk.text.upper()

            text = re.sub(
                r"[^A-Z0-9 ]",
                "",
                text
            ).strip()

            text = text.replace(" ", "_")

            if len(text) > 3:

                chunks.append(text)

        if len(chunks) >= 2:

            final_tag = f"{chunks[0]}_{chunks[1]}"

        elif len(chunks) == 1:

            final_tag = f"{chunks[0]}_EVENT"

        else:

            final_tag = "GENERAL_EVENT"

    final_tag = re.sub(
        r"_+",
        "_",
        final_tag
    )

    final_tag = final_tag.strip("_")

    words = final_tag.split("_")

    words = words[:4]

    final_tag = "_".join(words)

    return final_tag[:60]

def generate_event_summary(cluster_df):

    summaries = []

    for _, row in cluster_df.iterrows():

        summary = str(
            row["article_summary"]
        ).strip()

        if len(summary) > 30:

            summaries.append(summary)

    summaries = summaries[:5]

    final_summary = " ".join(summaries)

    return final_summary[:3000]

instruction = (
    "Represent this news article "
    "for retrieving related "
    "news articles: "
)

texts = [

    instruction + text

    for text in df["event_text"]

]

print("\nGenerating embeddings...\n")

embeddings = model.encode(

    texts,

    normalize_embeddings=True,

    batch_size=16,

    show_progress_bar=True

)

print("\nClustering articles...\n")

clustering = DBSCAN(

    eps=1 - SIMILARITY_THRESHOLD,

    min_samples=MIN_CLUSTER_SIZE,

    metric="cosine"

)

cluster_labels = clustering.fit_predict(
    embeddings
)

df["cluster_id"] = cluster_labels

used_article_indices = set()

event_rows = []

event_counter = 1

cluster_sizes = (

    df[df["cluster_id"] != -1]

    .groupby("cluster_id")

    .size()

    .sort_values(ascending=False)

)

related_clusters = (
    cluster_sizes.index.tolist()
)

for cluster_id in tqdm(related_clusters):

    cluster_df = df[
        df["cluster_id"] == cluster_id
    ].copy()

    cluster_df = cluster_df[
        ~cluster_df.index.isin(
            used_article_indices
        )
    ]

    if len(cluster_df) < 2:
        continue

    cluster_df = cluster_df.head(
        MAX_CLUSTER_SIZE
    )

    cluster_df = cluster_df.sort_values(
        by="headline_english"
    )

    event_tag = generate_event_tag(
        cluster_df
    )

    event_summary = generate_event_summary(
        cluster_df
    )

    cluster_indices = cluster_df.index.tolist()

    cluster_embeddings = embeddings[
        cluster_indices
    ]

    sim_matrix = cosine_similarity(
        cluster_embeddings
    )

    avg_similarity = np.mean(
        sim_matrix
    )

    avg_similarity = round(
        float(avg_similarity),
        4
    )

    event_id = (
        f"EV{event_counter:05d}"
    )

    row = {

        "event_id":
            event_id,

        "event_tag":
            event_tag,

        "event_type":
            "RELATED_ARTICLES_EVENT",

        "article_nos":
            " | ".join(

                cluster_df[
                    "article_no"
                ]
                .astype(str)
                .tolist()

            ),

        "hash_ids":
            " | ".join(

                cluster_df[
                    "hash_id"
                ]
                .astype(str)
                .tolist()

            ),

        "headline_english":
            " | ".join(

                cluster_df[
                    "headline_english"
                ]
                .astype(str)
                .tolist()

            ),

        "published_dates":
            " | ".join(

                cluster_df[
                    "published_date"
                ]
                .astype(str)
                .tolist()

            ),

        "published_times":
            " | ".join(

                cluster_df[
                    "published_time"
                ]
                .astype(str)
                .tolist()

            ),

        "event_summary":
            event_summary,

        "similarity_score":
            avg_similarity,

        "total_articles":
            len(cluster_df)

    }

    event_rows.append(row)

    for idx in cluster_df.index:

        used_article_indices.add(idx)

    event_counter += 1

remaining_df = df[
    ~df.index.isin(
        used_article_indices
    )
]

print(
    f"\nRemaining individual articles: "
    f"{len(remaining_df)}"
)

for _, row_data in tqdm(

    remaining_df.iterrows(),

    total=len(remaining_df)

):

    single_df = pd.DataFrame(
        [row_data]
    )

    event_tag = generate_event_tag(
        single_df
    )

    event_summary = str(
        row_data[
            "article_summary"
        ]
    )

    event_id = (
        f"EV{event_counter:05d}"
    )

    row = {

        "event_id":
            event_id,

        "event_tag":
            event_tag,

        "event_type":
            "INDIVIDUAL_EVENT",

        "article_nos":
            str(
                row_data[
                    "article_no"
                ]
            ),

        "hash_ids":
            str(
                row_data[
                    "hash_id"
                ]
            ),

        "headline_english":
            str(
                row_data[
                    "headline_english"
                ]
            ),

        "published_dates":
            str(
                row_data[
                    "published_date"
                ]
            ),

        "published_times":
            str(
                row_data[
                    "published_time"
                ]
            ),

        "event_summary":
            event_summary,

        "similarity_score":
            1.0,

        "total_articles":
            1

    }

    event_rows.append(row)

    event_counter += 1

events_df = pd.DataFrame(
    event_rows
)

events_df["sort_order"] = events_df[
    "event_type"
].map({

    "RELATED_ARTICLES_EVENT": 0,

    "INDIVIDUAL_EVENT": 1

})

events_df = events_df.sort_values(

    by=[

        "sort_order",
        "total_articles",
        "similarity_score"

    ],

    ascending=[True, False, False]

)

events_df = events_df.reset_index(
    drop=True
)

events_df.drop(
    columns=["sort_order"],
    inplace=True
)

events_df["event_id"] = [

    f"EV{str(i+1).zfill(5)}"

    for i in range(len(events_df))

]

final_total_articles = events_df[
    "total_articles"
].sum()

print(
    f"\nTOTAL ARTICLES REPRESENTED: "
    f"{final_total_articles}"
)

print(
    f"\nORIGINAL ARTICLES: "
    f"{ORIGINAL_TOTAL_ARTICLES}"
)

if final_total_articles == ORIGINAL_TOTAL_ARTICLES:

    print(
        "\nALL ARTICLES SUCCESSFULLY INCLUDED"
    )

else:

    print(
        "\nWARNING: ARTICLE COUNT MISMATCH"
    )

events_df.to_csv(

    OUTPUT_CSV,

    index=False,

    encoding="utf-8-sig"

)

print("\nCSV GENERATED SUCCESSFULLY")

print(f"\nTOTAL EVENTS: {len(events_df)}")

print(f"\nOUTPUT FILE:\n{OUTPUT_CSV}")

print("\nSAMPLE OUTPUT:\n")

print(events_df.head())