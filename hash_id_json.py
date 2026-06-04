import pandas as pd
import json

ARTICLES_CSV = r"C:\ai_project\mzamin_article_nlp_pg.csv"

EVENTS_CSV = r"C:\ai_project\articles_events_tags_all_processed_new_deep1.csv"

OUTPUT_JSON = r"C:\ai_project\event_articles.json"

OUTPUT_CSV = r"C:\ai_project\event_articles.csv"

articles_df = pd.read_csv(
    ARTICLES_CSV,
    dtype=str,
    low_memory=False
).fillna("")

events_df = pd.read_csv(
    EVENTS_CSV,
    dtype=str,
    low_memory=False
).fillna("")

articles_df = articles_df.drop_duplicates(
    subset=["hash_id"]
)

article_lookup = articles_df.set_index(
    "hash_id"
).to_dict("index")

records = []

missing_hash_ids = []

for _, event_row in events_df.iterrows():

    event_id = str(
        event_row.get(
            "event_id",
            ""
        )
    ).strip()

    hash_ids_text = str(
        event_row.get(
            "hash_ids",
            ""
        )
    )

    hash_ids = [

        h.strip()

        for h in hash_ids_text.split("|")

        if h.strip()

    ]

    for hash_id in hash_ids:

        article = article_lookup.get(
            hash_id
        )

        if article is None:

            missing_hash_ids.append(
                hash_id
            )

            continue

        records.append({

            "hash_id":
                hash_id,

            "article_url":
                article.get(
                    "article_url",
                    ""
                ),

            "headline_english":
                article.get(
                    "headline_english",
                    ""
                ),

            "article_summary":
                article.get(
                    "article_summary",
                    ""
                ),

            "event_id":
                event_id

        })

event_articles_df = pd.DataFrame(
    records
)

event_articles_df = event_articles_df.sort_values(

    by=[

        "event_id",
        "hash_id"

    ]

).reset_index(
    drop=True
)

event_articles_df.to_csv(

    OUTPUT_CSV,

    index=False,

    encoding="utf-8-sig"

)

with open(

    OUTPUT_JSON,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        event_articles_df.to_dict(
            orient="records"
        ),

        f,

        ensure_ascii=False,

        indent=4

    )

print(
    f"\nCSV Saved:\n{OUTPUT_CSV}"
)

print(
    f"\nJSON Saved:\n{OUTPUT_JSON}"
)

print(
    f"\nTotal Records: "
    f"{len(event_articles_df)}"
)

print(
    f"\nUnique Events: "
    f"{event_articles_df['event_id'].nunique()}"
)

print(
    f"\nUnique Articles: "
    f"{event_articles_df['hash_id'].nunique()}"
)

if missing_hash_ids:

    pd.DataFrame({

        "missing_hash_id":
            missing_hash_ids

    }).to_csv(

        r"C:\ai_project\missing_hash_ids.csv",

        index=False,

        encoding="utf-8-sig"

    )

    print(
        "\nMissing hash IDs saved to:"
    )

    print(
        r"C:\ai_project\missing_hash_ids.csv"
    )

print("\nDone.")