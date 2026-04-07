import json
import re
from pathlib import Path

import nltk
from num2words import num2words
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


rawFile = Path("data/reviews_raw.jsonl")
cleanFile = Path("data/reviews_clean.jsonl")

minWords = 3


nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

stopWords = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def changeNumbers(text):
    def replaceNum(match):
        try:
            return " " + num2words(int(match.group())) + " "
        except:
            return " "
    return re.sub(r"\d+", replaceNum, text)


def cleanText(text):
    if text is None:
        return ""

    text = text.lower()
    text = changeNumbers(text)

    # remove emojis and weird symbols
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # keep only letters, numbers, and spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    # remove stop words
    words = [word for word in words if word not in stopWords]

    # lemmatize words
    words = [lemmatizer.lemmatize(word) for word in words]

    text = " ".join(words)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    reviews = []

    with open(rawFile, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                reviews.append(json.loads(line))

    cleanedReviews = []
    seen = set()

    for review in reviews:
        cleaned = cleanText(review.get("content", ""))

        # remove empty reviews
        if cleaned == "":
            continue

        # remove very short reviews
        if len(cleaned.split()) < minWords:
            continue

        # remove duplicates
        if cleaned in seen:
            continue

        seen.add(cleaned)

        cleanedReview = {
            "app_name": review.get("app_name"),
            "review_index": review.get("review_index"),
            "id": review.get("id"),
            "content": cleaned
        }

        cleanedReviews.append(cleanedReview)

    cleanFile.parent.mkdir(parents=True, exist_ok=True)

    with open(cleanFile, "w", encoding="utf-8") as f:
        for review in cleanedReviews:
            f.write(json.dumps(review, ensure_ascii=False) + "\n")

    print("Raw reviews loaded:", len(reviews))
    print("Cleaned reviews saved:", len(cleanedReviews))


if __name__ == "__main__":
    main()