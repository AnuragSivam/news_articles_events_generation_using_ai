import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
import spacy

INPUT_CSV = r"C:\ai_project\mzamin_articles_processed.csv"
OUTPUT_CSV = r"C:\ai_project\articles_events_tags_all_processed.csv"

SIMILARITY_THRESHOLD = 0.80
MIN_CLUSTER_SIZE = 2

df = pd.read_csv(INPUT_CSV)

df = df.fillna("")

nlp = spacy.load("en_core_web_sm")

def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()

def create_event_text(row):

    headline = clean_text(
        row.get("headline_english", "")
    )

    summary = clean_text(
        row.get("summary_english", "")
    )

    keywords = clean_text(
        row.get("keywords", "")
    )

    body = clean_text(
        row.get("body_content_english", "")
    )

    body = body[:4000]

    combined = f"""
    HEADLINE:
    {headline}

    SUMMARY:
    {summary}

    KEYWORDS:
    {keywords}

    BODY:
    {body}
    """

    return combined.strip()

df["event_text"] = df.apply(
    create_event_text,
    axis=1
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(
    df["event_text"].tolist(),
    batch_size=16,
    show_progress_bar=True
)

clustering = DBSCAN(
    eps=1 - SIMILARITY_THRESHOLD,
    min_samples=MIN_CLUSTER_SIZE,
    metric="cosine"
)

cluster_labels = clustering.fit_predict(
    embeddings
)

df["cluster_id"] = cluster_labels

def generate_event_tag(cluster_df):

    headline_text = " ".join(
        cluster_df["headline_english"]
        .astype(str)
        .tolist()
    )

    summary_text = " ".join(
        cluster_df["summary_english"]
        .astype(str)
        .tolist()
    )

    text = headline_text + " " + summary_text

    doc = nlp(text)

    entities = []

    for ent in doc.ents:

        if ent.label_ in [
            "ORG",
            "GPE",
            "EVENT",
            "PERSON",
            "NORP",
            "FAC"
        ]:

            entity = ent.text.strip()

            entity = re.sub(
                r"[^A-Za-z0-9 ]",
                "",
                entity
            )

            entity = entity.upper()

            if len(entity) > 3:
                entities.append(entity)

    entities = list(dict.fromkeys(entities))

    important_nouns = []

    for token in doc:

        if token.pos_ in [
            "NOUN",
            "PROPN"
        ]:

            word = token.lemma_.upper()

            if (
                len(word) > 3
                and word not in [
                    "SAID",
                    "WILL",
                    "YEAR",
                    "TIME",
                    "PEOPLE",
                    "COUNTRY",
                    "GOVERNMENT",
                    "BANGLADESH"
                ]
            ):

                important_nouns.append(word)

    noun_counts = (
        pd.Series(important_nouns)
        .value_counts()
    )

    top_nouns = (
        noun_counts
        .head(3)
        .index
        .tolist()
    )

    final_parts = []

    for entity in entities[:2]:

        final_parts.append(entity)

    for noun in top_nouns:

        if noun not in final_parts:

            final_parts.append(noun)

    if len(final_parts) == 0:

        headline = str(
            cluster_df.iloc[0][
                "headline_english"
            ]
        )

        headline = re.sub(
            r"[^A-Za-z0-9 ]",
            "",
            headline
        )

        headline = (
            headline.upper()
            .replace(" ", "_")
        )

        return headline[:120]

    final_tag = "_".join(
        final_parts[:4]
    )

    final_tag = re.sub(
        r"_+",
        "_",
        final_tag
    )

    final_tag = final_tag.strip("_")

    return final_tag[:120]

event_rows = []

cluster_sizes = (
    df[df["cluster_id"] != -1]
    .groupby("cluster_id")
    .size()
    .sort_values(ascending=False)
)

related_clusters = (
    cluster_sizes.index.tolist()
)

event_counter = 1

for cluster_id in tqdm(related_clusters):

    cluster_df = df[
        df["cluster_id"] == cluster_id
    ]

    event_tag = generate_event_tag(
        cluster_df
    )

    places = []

    for txt in cluster_df[
        "summary_english"
    ].tolist():

        doc = nlp(txt)

        for ent in doc.ents:

            if ent.label_ in [
                "GPE",
                "LOC"
            ]:

                places.append(
                    ent.text
                )

    if len(places) == 0:

        place_text = "UNKNOWN"

    else:

        place_text = " | ".join(
            list(
                dict.fromkeys(
                    places
                )
            )
        )

    event_id = (
        f"EV{event_counter:05d}"
    )

    row = {

        "event_id": event_id,

        "event_tag": event_tag,

        "event_type": "RELATED_ARTICLES_EVENT",

        "article_nos": " | ".join(
            cluster_df[
                "article_no"
            ]
            .astype(str)
            .tolist()
        ),

        "hash_ids": " | ".join(
            cluster_df[
                "hash_id"
            ]
            .astype(str)
            .tolist()
        ),

        "headlines": " | ".join(
            cluster_df[
                "headline_english"
            ]
            .astype(str)
            .tolist()
        ),

        "published_dates": " | ".join(
            cluster_df[
                "published_date"
            ]
            .astype(str)
            .tolist()
        ),

        "published_times": " | ".join(
            cluster_df[
                "published_time"
            ]
            .astype(str)
            .tolist()
        ),

        "places": place_text,

        "total_articles": len(
            cluster_df
        )
    }

    event_rows.append(row)

    event_counter += 1

individual_df = df[
    df["cluster_id"] == -1
]

for _, row_data in tqdm(
    individual_df.iterrows(),
    total=len(individual_df)
):

    single_df = pd.DataFrame(
        [row_data]
    )

    event_tag = generate_event_tag(
        single_df
    )

    places = []

    doc = nlp(
        str(
            row_data[
                "summary_english"
            ]
        )
    )

    for ent in doc.ents:

        if ent.label_ in [
            "GPE",
            "LOC"
        ]:

            places.append(
                ent.text
            )

    if len(places) == 0:

        place_text = "UNKNOWN"

    else:

        place_text = " | ".join(
            list(
                dict.fromkeys(
                    places
                )
            )
        )

    event_id = (
        f"EV{event_counter:05d}"
    )

    row = {

        "event_id": event_id,

        "event_tag": event_tag,

        "event_type": "INDIVIDUAL_EVENT",

        "article_nos": str(
            row_data[
                "article_no"
            ]
        ),

        "hash_ids": str(
            row_data[
                "hash_id"
            ]
        ),

        "headlines": str(
            row_data[
                "headline_english"
            ]
        ),

        "published_dates": str(
            row_data[
                "published_date"
            ]
        ),

        "published_times": str(
            row_data[
                "published_time"
            ]
        ),

        "places": place_text,

        "total_articles": 1
    }

    event_rows.append(row)

    event_counter += 1

events_df = pd.DataFrame(
    event_rows
)

events_df = events_df.reset_index(
    drop=True
)

events_df["event_id"] = [
    f"EV{str(i+1).zfill(5)}"
    for i in range(len(events_df))
]

events_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

print(events_df.head())

print("\nTOTAL EVENTS:")
print(len(events_df))

print("\nOUTPUT SAVED:")
print(OUTPUT_CSV)