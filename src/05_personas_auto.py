import json
import os
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


# Config

INPUT_REVIEWS_FILE = "data/reviews_clean.jsonl"
OUTPUT_GROUPS_FILE = "data/review_groups_auto.json"
OUTPUT_PERSONAS_FILE = "personas/personas_auto.json"
OUTPUT_PROMPT_FILE = "prompts/prompt_auto.json"

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Fixed number of groups/personas
FIXED_K = 8

# Max number of example reviews kept per group in output json
MAX_EXAMPLE_REVIEWS = 15

# Max number of review indexes shown to the LLM per cluster
MAX_REVIEW_INDEXES_FOR_PROMPT = 15

# Max number of example reviews shown to the LLM per cluster
MAX_EXAMPLES_FOR_PROMPT = 8

# Number of keywords to extract per cluster
TOP_KEYWORDS_PER_CLUSTER = 6

RANDOM_STATE = 42


# File Helpers

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_json(path: str, data: Any) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_reviews_jsonl(path: str) -> list[dict]:
    reviews = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reviews.append(json.loads(line))
    return reviews


# Review Field Access

def get_review_index(review_obj: dict) -> str:
    if "review_index" in review_obj:
        return f"rev_{int(review_obj['review_index'])}"
    raise KeyError(f"Could not find review_index field in review object: {review_obj}")


def get_review_text(review_obj: dict) -> str:
    for key in ["content", "review", "text", "cleaned_review", "review_text"]:
        if key in review_obj and review_obj[key]:
            return str(review_obj[key])
    raise KeyError(f"Could not find review text field in review object: {review_obj}")


# LLM Helpers

def strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def extract_json_object(text: str) -> dict:
    text = strip_code_fences(text).strip()
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON object in model response.")

    candidate = text[start:end + 1]
    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        print("\n--- RAW MODEL RESPONSE START ---")
        print(text[:5000])
        print("--- RAW MODEL RESPONSE END ---\n")
        raise ValueError(f"Model returned invalid JSON even after cleanup: {e}")


def groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set in the environment.")
    return Groq(api_key=api_key)


def call_llm_json(client: Groq, prompt: str, max_retries: int = 3) -> dict:
    last_error = None

    for attempt in range(1, max_retries + 1):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.1,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are assisting with software requirements engineering. "
                        "Return only one valid JSON object. "
                        "Do not include markdown fences. "
                        "Do not include comments. "
                        "Do not include explanatory text before or after the JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        content = response.choices[0].message.content or ""

        try:
            return extract_json_object(content)
        except Exception as e:
            last_error = e
            print(f"JSON parse failed on attempt {attempt}/{max_retries}: {e}")
            prompt = (
                prompt
                + "\n\nIMPORTANT: Your previous answer was invalid. "
                  "Return only one valid JSON object. "
                  "Do not use markdown. "
                  "Do not add explanation text."
            )

    raise ValueError(f"Failed to get valid JSON after {max_retries} attempts: {last_error}")


# Embedding + Clustering

def embed_reviews(texts: list[str]) -> np.ndarray:
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings


def cluster_reviews(embeddings: np.ndarray, k: int) -> np.ndarray:
    print(f"Clustering reviews into {k} groups ...")
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    return labels


# Cluster Summaries

def extract_cluster_keywords(cluster_texts: list[str], top_n: int) -> list[str]:
    if not cluster_texts:
        return []

    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    matrix = vectorizer.fit_transform(cluster_texts)
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    features = np.array(vectorizer.get_feature_names_out())

    top_indices = scores.argsort()[::-1][:top_n]
    keywords = [features[i] for i in top_indices if scores[i] > 0]
    return keywords


def choose_example_reviews_for_cluster(
    cluster_indices: list[int],
    embeddings: np.ndarray,
    max_examples: int
) -> list[int]:
    if not cluster_indices:
        return []

    cluster_vectors = embeddings[cluster_indices]
    centroid = cluster_vectors.mean(axis=0)

    distances = []
    for idx in cluster_indices:
        dist = np.linalg.norm(embeddings[idx] - centroid)
        distances.append((dist, idx))

    distances.sort(key=lambda x: x[0])
    chosen = [idx for _, idx in distances[:max_examples]]
    return chosen


def build_initial_groups(
    reviews: list[dict],
    texts: list[str],
    embeddings: np.ndarray,
    labels: np.ndarray
) -> list[dict]:
    grouped_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        grouped_indices[int(label)].append(idx)

    groups = []
    sorted_cluster_ids = sorted(grouped_indices.keys())

    for new_num, cluster_id in enumerate(sorted_cluster_ids, start=1):
        indices = grouped_indices[cluster_id]

        review_indexes = [get_review_index(reviews[i]) for i in indices]
        cluster_texts = [texts[i] for i in indices]

        example_indices = choose_example_reviews_for_cluster(
            cluster_indices=indices,
            embeddings=embeddings,
            max_examples=MAX_EXAMPLE_REVIEWS
        )
        example_reviews = [texts[i] for i in example_indices]

        keywords = extract_cluster_keywords(cluster_texts, TOP_KEYWORDS_PER_CLUSTER)

        groups.append({
            "group_id": f"G{new_num}",
            "theme": "",
            "review_indexes": review_indexes,
            "example_reviews": example_reviews,
            "keywords": keywords,
        })

    return groups


# Prompt Builders

def build_group_labeling_prompt(groups: list[dict]) -> str:
    prompt_groups = []
    for g in groups:
        prompt_groups.append({
            "group_id": g["group_id"],
            "review_count": len(g["review_indexes"]),
            "keywords": g["keywords"][:TOP_KEYWORDS_PER_CLUSTER],
            "example_reviews": g["example_reviews"][:MAX_EXAMPLES_FOR_PROMPT],
            "review_indexes": g["review_indexes"][:MAX_REVIEW_INDEXES_FOR_PROMPT],
        })

    return f"""
You are helping with requirements engineering for the Calm app.

Task:
Given the following automatically clustered review groups, assign a short, meaningful theme to each group.

Rules:
- Keep the existing group_id exactly.
- Do not merge or split groups.
- Do not invent review indexes.
- Themes should describe a recurring user concern, user goal, pain point, or usage situation.
- Themes should be concise, specific, and useful for persona creation.
- Return JSON only.

Output format:
{{
  "groups": [
    {{
      "group_id": "G1",
      "theme": "short theme here"
    }}
  ]
}}

Cluster summaries:
{json.dumps(prompt_groups, indent=2, ensure_ascii=False)}
""".strip()


def build_persona_prompt(groups_for_personas: list[dict]) -> str:
    prompt_groups = []
    for g in groups_for_personas:
        prompt_groups.append({
            "group_id": g["group_id"],
            "theme": g["theme"],
            "review_count": len(g["review_indexes"]),
            "keywords": g["keywords"][:TOP_KEYWORDS_PER_CLUSTER],
            "example_reviews": g["example_reviews"][:MAX_EXAMPLES_FOR_PROMPT],
            "review_indexes": g["review_indexes"][:MAX_REVIEW_INDEXES_FOR_PROMPT],
        })

    return f"""
You are helping with requirements engineering for the Calm app.

Task:
Create exactly one persona for each review group below.

Rules:
- Personas must stay grounded in the review evidence.
- Do not invent unsupported biography details.
- Focus on realistic user context, goals, frustrations, and constraints.
- Each persona must clearly reference the review group it came from.
- Use the exact JSON schema below.
- "description" should be a short paragraph summarizing the persona.
- "derived_from_group" must match the group_id exactly.
- "evidence_reviews" must contain review indexes from that group only.
- Return JSON only.

Output format:
{{
  "personas": [
    {{
      "id": "P1",
      "name": "short persona name",
      "description": "short paragraph",
      "derived_from_group": "G1",
      "goals": ["goal 1", "goal 2"],
      "pain_points": ["pain point 1", "pain point 2"],
      "context": ["context 1", "context 2"],
      "constraints": ["constraint 1", "constraint 2"],
      "evidence_reviews": [12, 58, 201]
    }}
  ]
}}

Review groups:
{json.dumps(prompt_groups, indent=2, ensure_ascii=False)}
""".strip()


# Normalization Helpers

def normalize_persona(persona: dict, fallback_id: str, fallback_group_id: str) -> dict:
    evidence = persona.get("evidence_reviews", [])
    normalized_evidence = []
    for x in evidence:
        try:
            normalized_evidence.append(str(x).strip())
        except Exception:
            continue

    return {
        "id": str(persona.get("id", fallback_id)).strip(),
        "name": str(persona.get("name", "")).strip(),
        "description": str(persona.get("description", "")).strip(),
        "derived_from_group": str(persona.get("derived_from_group", fallback_group_id)).strip(),
        "goals": [str(x).strip() for x in persona.get("goals", []) if str(x).strip()],
        "pain_points": [str(x).strip() for x in persona.get("pain_points", []) if str(x).strip()],
        "context": [str(x).strip() for x in persona.get("context", []) if str(x).strip()],
        "constraints": [str(x).strip() for x in persona.get("constraints", []) if str(x).strip()],
        "evidence_reviews": normalized_evidence,
    }


def normalize_group_theme_result(result: dict) -> dict[str, str]:
    mapping = {}
    for item in result.get("groups", []):
        gid = str(item.get("group_id", "")).strip()
        theme = str(item.get("theme", "")).strip()
        if gid:
            mapping[gid] = theme
    return mapping


def count_review_assignments(groups: list[dict]) -> Counter:
    c = Counter()
    for g in groups:
        for rid in g.get("review_indexes", []):
            c[rid] += 1
    return c


# LLM Post-processing

def apply_llm_group_themes(client: Groq, groups: list[dict], prompt_log: dict) -> list[dict]:
    print("Generating group themes with the LLM ...")
    prompt = build_group_labeling_prompt(groups)
    prompt_log["task_4_1"]["group_label_prompt"] = prompt

    result = call_llm_json(client, prompt)
    theme_map = normalize_group_theme_result(result)

    for i, group in enumerate(groups, start=1):
        group["theme"] = theme_map.get(group["group_id"], f"User feedback cluster {i}")

    return groups


def generate_personas_from_groups(client: Groq, groups: list[dict], prompt_log: dict) -> list[dict]:
    print("Generating personas from clustered groups ...")
    prompt = build_persona_prompt(groups)
    prompt_log["task_4_2"]["persona_prompt"] = prompt

    result = call_llm_json(client, prompt)
    personas = result.get("personas", [])

    normalized = []
    for i, persona in enumerate(personas, start=1):
        fallback_group_id = groups[min(i - 1, len(groups) - 1)]["group_id"]
        normalized.append(normalize_persona(persona, f"P{i}", fallback_group_id))

    while len(normalized) < len(groups):
        idx = len(normalized) + 1
        group = groups[idx - 1]
        normalized.append({
            "id": f"P{idx}",
            "name": f"Persona for {group['theme']}",
            "description": f"This persona represents users associated with the theme: {group['theme']}.",
            "derived_from_group": group["group_id"],
            "goals": [],
            "pain_points": [],
            "context": [],
            "constraints": [],
            "evidence_reviews": group["review_indexes"][:4],
        })

    normalized = normalized[:len(groups)]

    group_lookup = {g["group_id"]: g for g in groups}

    for i, persona in enumerate(normalized, start=1):
        persona["id"] = f"P{i}"

        if persona["derived_from_group"] not in group_lookup:
            persona["derived_from_group"] = groups[min(i - 1, len(groups) - 1)]["group_id"]

        valid_reviews = set(group_lookup[persona["derived_from_group"]]["review_indexes"])
        persona["evidence_reviews"] = [
            rid for rid in persona["evidence_reviews"] if rid in valid_reviews
        ][:4]

        if not persona["evidence_reviews"]:
            persona["evidence_reviews"] = group_lookup[persona["derived_from_group"]]["review_indexes"][:4]

    return normalized


# Main

def main() -> None:
    reviews = load_reviews_jsonl(INPUT_REVIEWS_FILE)
    print(f"Loaded {len(reviews)} cleaned reviews.")

    texts = [get_review_text(r) for r in reviews]
    embeddings = embed_reviews(texts)

    labels = cluster_reviews(embeddings, FIXED_K)
    groups = build_initial_groups(reviews, texts, embeddings, labels)

    prompt_log = {
        "model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "task_4_1": {
            "clustering_method": "SentenceTransformer embeddings + KMeans",
            "selected_k": FIXED_K,
            "group_label_prompt": None,
        },
        "task_4_2": {
            "persona_prompt": None
        }
    }

    client = groq_client()
    groups = apply_llm_group_themes(client, groups, prompt_log)
    personas = generate_personas_from_groups(client, groups, prompt_log)

    review_groups_payload = {"groups": groups}
    personas_payload = {"personas": personas}

    save_json(OUTPUT_GROUPS_FILE, review_groups_payload)
    save_json(OUTPUT_PERSONAS_FILE, personas_payload)
    save_json(OUTPUT_PROMPT_FILE, prompt_log)

    print(f"Saved: {OUTPUT_GROUPS_FILE}")
    print(f"Saved: {OUTPUT_PERSONAS_FILE}")
    print(f"Saved: {OUTPUT_PROMPT_FILE}")

    assignment_counts = count_review_assignments(groups)
    duplicates = [rid for rid, count in assignment_counts.items() if count > 1]
    unassigned = [
        get_review_index(r)
        for r in reviews
        if assignment_counts.get(get_review_index(r), 0) == 0
    ]

    print("\nSanity check:")
    print(f"- Selected cluster count: {FIXED_K}")
    print(f"- Final group count: {len(groups)}")
    print(f"- Persona count: {len(personas)}")
    print(f"- Duplicate review assignments: {len(duplicates)}")
    print(f"- Unassigned reviews: {len(unassigned)}")

    for group in groups:
        print(f"  {group['group_id']} | {group['theme']} | {len(group['review_indexes'])} reviews")


if __name__ == "__main__":
    main()