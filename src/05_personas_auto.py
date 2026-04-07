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


# input/output paths
INPUT_REVIEWS_FILE = "data/reviews_clean.jsonl"
OUTPUT_GROUPS_FILE = "data/review_groups_auto.json"
OUTPUT_PERSONAS_FILE = "personas/personas_auto.json"
OUTPUT_PROMPT_FILE = "prompts/prompt_auto.json"

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# how many clusters/personas to generate
FIXED_K = 8

# how many example reviews to store in the output json per group
MAX_EXAMPLE_REVIEWS = 50

# how many review indexes we send to the LLM per cluster (keeps the prompt short)
MAX_REVIEW_INDEXES_FOR_PROMPT = 15

# how many example reviews we send to the LLM per cluster
MAX_EXAMPLES_FOR_PROMPT = 8

# how many keywords to pull per cluster
TOP_KEYWORDS_PER_CLUSTER = 6

RANDOM_STATE = 42


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


def get_review_index(review_obj: dict) -> str:
    if "review_index" in review_obj:
        return f"rev_{int(review_obj['review_index'])}"
    raise KeyError(f"no review_index field found in: {review_obj}")


def get_review_text(review_obj: dict) -> str:
    # try a few common field names for the review text
    for key in ["content", "review", "text", "cleaned_review", "review_text"]:
        if key in review_obj and review_obj[key]:
            return str(review_obj[key])
    raise KeyError(f"no review text field found in: {review_obj}")


def strip_code_fences(text: str) -> str:
    # remove markdown code blocks if the LLM wraps its response in them
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
    # normalize fancy quotes that sometimes appear in LLM output
    text = text.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # if full parse fails, try to find and extract just the JSON object
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("could not find a JSON object in the model response")

    candidate = text[start:end + 1]
    candidate = re.sub(r",\s*([\]}])", r"\1", candidate)  # remove trailing commas

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        print("\nraw model response:")
        print(text[:5000])
        raise ValueError(f"model returned invalid JSON even after cleanup: {e}")


def groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY environment variable is not set")
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
            print(f"  JSON parse failed on attempt {attempt}/{max_retries}: {e}")
            # remind the model to return valid JSON on the next attempt
            prompt = (
                prompt
                + "\n\nIMPORTANT: Your previous answer was invalid. "
                  "Return only one valid JSON object. "
                  "Do not use markdown. "
                  "Do not add explanation text."
            )

    raise ValueError(f"failed to get valid JSON after {max_retries} attempts: {last_error}")


def embed_reviews(texts: list[str]) -> np.ndarray:
    print(f"loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return embeddings


def cluster_reviews(embeddings: np.ndarray, k: int) -> np.ndarray:
    print(f"clustering reviews into {k} groups...")
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
    return kmeans.fit_predict(embeddings)


def extract_cluster_keywords(cluster_texts: list[str], top_n: int) -> list[str]:
    if not cluster_texts:
        return []
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(cluster_texts)
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    features = np.array(vectorizer.get_feature_names_out())
    top_indices = scores.argsort()[::-1][:top_n]
    return [features[i] for i in top_indices if scores[i] > 0]


def choose_example_reviews_for_cluster(
    cluster_indices: list[int],
    embeddings: np.ndarray,
    max_examples: int
) -> list[int]:
    # pick the reviews closest to the cluster centroid as examples
    if not cluster_indices:
        return []
    cluster_vectors = embeddings[cluster_indices]
    centroid = cluster_vectors.mean(axis=0)
    distances = [(np.linalg.norm(embeddings[idx] - centroid), idx) for idx in cluster_indices]
    distances.sort(key=lambda x: x[0])
    return [idx for _, idx in distances[:max_examples]]


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
    for new_num, cluster_id in enumerate(sorted(grouped_indices.keys()), start=1):
        indices = grouped_indices[cluster_id]
        review_indexes = [get_review_index(reviews[i]) for i in indices]
        cluster_texts = [texts[i] for i in indices]
        example_indices = choose_example_reviews_for_cluster(indices, embeddings, MAX_EXAMPLE_REVIEWS)
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


def build_persona_prompt_single(group: dict, persona_id: str) -> str:
    # one prompt per group so we never hit the token limit
    prompt_group = {
        "group_id": group["group_id"],
        "theme": group["theme"],
        "review_count": len(group["review_indexes"]),
        "keywords": group["keywords"][:TOP_KEYWORDS_PER_CLUSTER],
        "example_reviews": group["example_reviews"][:MAX_EXAMPLES_FOR_PROMPT],
        "review_indexes": group["review_indexes"][:MAX_REVIEW_INDEXES_FOR_PROMPT],
    }

    return f"""
You are helping with requirements engineering for the Calm app.

Task:
Create exactly one persona for the review group below.

Rules:
- The persona must stay grounded in the review evidence.
- Do not invent unsupported biography details.
- Focus on realistic user context, goals, frustrations, and constraints.
- Use the exact JSON schema below.
- "description" should be a short paragraph summarizing the persona.
- "derived_from_group" must be "{group["group_id"]}".
- "evidence_reviews" must contain review indexes from the group only.
- Return JSON only.

Output format:
{{
  "id": "{persona_id}",
  "name": "short persona name",
  "description": "short paragraph",
  "derived_from_group": "{group["group_id"]}",
  "goals": ["goal 1", "goal 2"],
  "pain_points": ["pain point 1", "pain point 2"],
  "context": ["context 1", "context 2"],
  "constraints": ["constraint 1", "constraint 2"],
  "evidence_reviews": ["rev_12", "rev_58"]
}}

Review group:
{json.dumps(prompt_group, indent=2, ensure_ascii=False)}
""".strip()


def normalize_persona(persona: dict, fallback_id: str, fallback_group_id: str) -> dict:
    # clean up the persona fields and make sure nothing is missing
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


def apply_llm_group_themes(client: Groq, groups: list[dict], prompt_log: dict) -> list[dict]:
    print("generating group themes with the LLM...")
    prompt = build_group_labeling_prompt(groups)
    prompt_log["task_4_1"]["group_label_prompt"] = prompt

    result = call_llm_json(client, prompt)
    theme_map = normalize_group_theme_result(result)

    for i, group in enumerate(groups, start=1):
        group["theme"] = theme_map.get(group["group_id"], f"user feedback cluster {i}")

    return groups


def generate_personas_from_groups(client: Groq, groups: list[dict], prompt_log: dict) -> list[dict]:
    # call the LLM once per group instead of all at once to avoid token limit issues
    print("generating personas from clustered groups (one per group)...")
    prompt_log["task_4_2"]["persona_prompts"] = []

    group_lookup = {g["group_id"]: g for g in groups}
    normalized = []

    for i, group in enumerate(groups, start=1):
        persona_id = f"P{i}"
        print(f"  generating persona {persona_id} for {group['group_id']}: {group['theme']}")

        prompt = build_persona_prompt_single(group, persona_id)
        prompt_log["task_4_2"]["persona_prompts"].append({
            "group_id": group["group_id"],
            "prompt": prompt,
        })

        try:
            result = call_llm_json(client, prompt)
            # the LLM returns a single object this time, not a list
            persona = normalize_persona(result, persona_id, group["group_id"])
        except Exception as e:
            print(f"  failed to generate persona for {group['group_id']}: {e}, using placeholder")
            persona = {
                "id": persona_id,
                "name": f"persona for {group['theme']}",
                "description": f"represents users associated with: {group['theme']}",
                "derived_from_group": group["group_id"],
                "goals": [],
                "pain_points": [],
                "context": [],
                "constraints": [],
                "evidence_reviews": group["review_indexes"][:4],
            }

        persona["id"] = persona_id
        persona["derived_from_group"] = group["group_id"]

        # only keep evidence reviews that actually belong to this group
        valid_reviews = set(group["review_indexes"])
        persona["evidence_reviews"] = [
            rid for rid in persona["evidence_reviews"] if rid in valid_reviews
        ][:4]

        if not persona["evidence_reviews"]:
            persona["evidence_reviews"] = group["review_indexes"][:4]

        normalized.append(persona)

    return normalized


def main() -> None:
    reviews = load_reviews_jsonl(INPUT_REVIEWS_FILE)
    print(f"loaded {len(reviews)} cleaned reviews")

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

    save_json(OUTPUT_GROUPS_FILE, {"groups": groups})
    save_json(OUTPUT_PERSONAS_FILE, {"personas": personas})
    save_json(OUTPUT_PROMPT_FILE, prompt_log)

    print(f"saved: {OUTPUT_GROUPS_FILE}")
    print(f"saved: {OUTPUT_PERSONAS_FILE}")
    print(f"saved: {OUTPUT_PROMPT_FILE}")

    # quick sanity check on the output
    assignment_counts = count_review_assignments(groups)
    duplicates = [rid for rid, count in assignment_counts.items() if count > 1]
    unassigned = [
        get_review_index(r)
        for r in reviews
        if assignment_counts.get(get_review_index(r), 0) == 0
    ]

    print("\nsanity check:")
    print(f"  clusters: {FIXED_K}, groups saved: {len(groups)}, personas saved: {len(personas)}")
    print(f"  duplicate review assignments: {len(duplicates)}")
    print(f"  unassigned reviews: {len(unassigned)}")
    for group in groups:
        print(f"  {group['group_id']} | {group['theme']} | {len(group['review_indexes'])} reviews")


if __name__ == "__main__":
    main()