import json
from pathlib import Path

from google_play_scraper import reviews_all, Sort

# Followed the template from our Lab 2 and apadted to collect reviews for the Calm app.
appId = "com.calm.android"
appName = "Calm"
lang = "en"
country = "ca"
reviewLimit = 5000

rawOutput = Path("data/reviews_raw.jsonl")


def main():
    rawOutput.parent.mkdir(parents=True, exist_ok=True)

    print("Collecting reviews...")
    rawReviews = reviews_all(
        appId,
        lang=lang,
        country=country,
        sort=Sort.NEWEST,
        sleep_milliseconds=0,
    )

    rawReviews = rawReviews[:reviewLimit]

    print("Writing reviews_raw.jsonl...")
    with open(rawOutput, "w", encoding="utf-8") as f:
        for i, review in enumerate(rawReviews, start=1):
            reviewRecord = {
                "app_name": appName,
                "review_index": i,
                "content": review.get("content", ""),
                "score": review.get("score"),
    "thumbsUpCount": review.get("thumbsUpCount"),
                "id": review.get("reviewId", f"calm_{i}")
            }
            f.write(json.dumps(reviewRecord, ensure_ascii=False) + "\n")

    print(f"Done. Saved {len(rawReviews)} reviews to {rawOutput}")


if __name__ == "__main__":
    main()