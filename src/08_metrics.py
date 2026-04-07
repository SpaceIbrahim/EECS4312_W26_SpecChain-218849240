import argparse
import json
import os
import re
from typing import Any


# Config

DEFAULT_REVIEWS_FILE = "data/reviews_clean.jsonl"

PIPELINE_CONFIGS = {
    "manual": {
        "groups_file": "data/review_groups_manual.json",
        "personas_file": "personas/personas_manual.json",
        "spec_file": "spec/spec_manual.md",
        "tests_file": "tests/tests_manual.json",
        "output_file": "metrics/metrics_manual.json",
    },
    "auto": {
        "groups_file": "data/review_groups_auto.json",
        "personas_file": "personas/personas_auto.json",
        "spec_file": "spec/spec_auto.md",
        "tests_file": "tests/tests_auto.json",
        "output_file": "metrics/metrics_auto.json",
    },
    "hybrid": {
        "groups_file": "data/review_groups_hybrid.json",
        "personas_file": "personas/personas_hybrid.json",
        "spec_file": "spec/spec_hybrid.md",
        "tests_file": "tests/tests_hybrid.json",
        "output_file": "metrics/metrics_hybrid.json",
    },
}

AMBIGUOUS_TERMS = {
    "fast",
    "easy",
    "better",
    "user-friendly",
    "user friendly",
    "seamless",
    "intuitive",
    "efficient",
    "quick",
    "simple",
    "smooth",
    "convenient",
    "effective",
    "reliable",
    "flexible",
    "responsive",
    "improved",
    "improve",
    "friendly",
}


# File Helpers

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


# Spec Parsing

def parse_spec_markdown(markdown_text: str) -> list[dict]:
    pattern = re.compile(
        r"# Requirement ID:\s*(?P<requirement_id>[^\n]+)\n"
        r"- Description:\s*\[(?P<description>.*?)\]\s*\n+"
        r"- Source Persona:\s*\[(?P<source_persona>.*?)\]\s*\n"
        r"- Traceability:\s*\[(?P<traceability>.*?)\]\s*\n"
        r"- Acceptance Criteria:\s*\[(?P<acceptance_criteria>.*?)\]",
        re.DOTALL
    )

    requirements = []
    for match in pattern.finditer(markdown_text):
        requirements.append({
            "requirement_id": match.group("requirement_id").strip(),
            "description": " ".join(match.group("description").split()),
            "source_persona": " ".join(match.group("source_persona").split()),
            "traceability": " ".join(match.group("traceability").split()),
            "acceptance_criteria": " ".join(match.group("acceptance_criteria").split()),
        })

    return requirements


def load_requirements(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return parse_spec_markdown(text)


# Metric Helpers

def round_ratio(value: float) -> float:
    return round(value, 2)


def collect_review_indexes_from_groups(groups: list[dict]) -> set[int]:
    covered = set()
    for group in groups:
        for rid in group.get("review_indexes", []):
            try:
                covered.add(int(rid))
            except Exception:
                continue
    return covered


def collect_review_indexes_from_personas(personas: list[dict]) -> set[int]:
    covered = set()
    for persona in personas:
        for rid in persona.get("evidence_reviews", []):
            try:
                covered.add(int(rid))
            except Exception:
                continue
    return covered


def requirement_is_traceable(req: dict) -> bool:
    has_persona = bool(req.get("source_persona", "").strip())
    has_traceability = bool(req.get("traceability", "").strip())
    return has_persona and has_traceability


def requirement_is_ambiguous(req: dict) -> bool:
    text = f"{req.get('description', '')} {req.get('acceptance_criteria', '')}".lower()
    normalized = re.sub(r"[^a-z0-9\s\-]", " ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    for term in AMBIGUOUS_TERMS:
        if term in normalized:
            return True
    return False


def compute_traceability_links(personas: list[dict], requirements: list[dict], tests: list[dict]) -> int:
    persona_to_group_links = 0
    for persona in personas:
        if persona.get("derived_from_group"):
            persona_to_group_links += 1

    req_to_persona_links = 0
    for req in requirements:
        if requirement_is_traceable(req):
            req_to_persona_links += 1

    test_to_requirement_links = 0
    for test in tests:
        if str(test.get("requirement_id", "")).strip():
            test_to_requirement_links += 1

    return persona_to_group_links + req_to_persona_links + test_to_requirement_links


def compute_metrics_for_pipeline(
    reviews_file: str,
    groups_file: str,
    personas_file: str,
    spec_file: str,
    tests_file: str,
) -> dict:
    reviews = load_reviews_jsonl(reviews_file)
    groups_payload = load_json(groups_file)
    personas_payload = load_json(personas_file)
    tests_payload = load_json(tests_file)
    requirements = load_requirements(spec_file)

    groups = groups_payload.get("groups", [])
    personas = personas_payload.get("personas", [])
    tests = tests_payload.get("tests", [])

    dataset_size = len(reviews)
    persona_count = len(personas)
    requirements_count = len(requirements)
    tests_count = len(tests)

    # Review coverage:
    # Prefer group coverage because groups represent the actual partition of the dataset.
    # If groups are missing, fall back to persona evidence_reviews.
    covered_reviews = collect_review_indexes_from_groups(groups)
    if not covered_reviews:
        covered_reviews = collect_review_indexes_from_personas(personas)

    review_coverage_ratio = (
        round_ratio(len(covered_reviews) / dataset_size) if dataset_size > 0 else 0.0
    )

    traceable_requirements = sum(1 for req in requirements if requirement_is_traceable(req))
    traceability_ratio = (
        round_ratio(traceable_requirements / requirements_count)
        if requirements_count > 0 else 0.0
    )

    tested_requirement_ids = {
        str(test.get("requirement_id", "")).strip()
        for test in tests
        if str(test.get("requirement_id", "")).strip()
    }
    testable_requirements = sum(
        1 for req in requirements
        if req["requirement_id"] in tested_requirement_ids
    )
    testability_rate = (
        round_ratio(testable_requirements / requirements_count)
        if requirements_count > 0 else 0.0
    )

    ambiguous_requirements = sum(1 for req in requirements if requirement_is_ambiguous(req))
    ambiguity_ratio = (
        round_ratio(ambiguous_requirements / requirements_count)
        if requirements_count > 0 else 0.0
    )

    traceability_links = compute_traceability_links(personas, requirements, tests)

    return {
        "dataset_size": dataset_size,
        "persona_count": persona_count,
        "requirements_count": requirements_count,
        "tests_count": tests_count,
        "traceability_links": traceability_links,
        "review_coverage_ratio": review_coverage_ratio,
        "traceability_ratio": traceability_ratio,
        "testability_rate": testability_rate,
        "ambiguity_ratio": ambiguity_ratio,
    }


# Summary Metrics

def build_metrics_summary() -> dict:
    summary = {}

    for pipeline_name, cfg in PIPELINE_CONFIGS.items():
        output_file = cfg["output_file"]
        if os.path.exists(output_file):
            summary[pipeline_name] = load_json(output_file)

    return {"pipelines": summary}


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute SpecChain metrics for manual, auto, or hybrid pipelines."
    )
    parser.add_argument(
        "--pipeline",
        choices=["manual", "auto", "hybrid", "all"],
        default="auto",
        help="Which pipeline metrics to compute."
    )
    parser.add_argument(
        "--reviews-file",
        default=DEFAULT_REVIEWS_FILE,
        help="Path to cleaned reviews jsonl file."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipelines = (
        ["manual", "auto", "hybrid"]
        if args.pipeline == "all"
        else [args.pipeline]
    )

    for pipeline_name in pipelines:
        cfg = PIPELINE_CONFIGS[pipeline_name]

        metrics = compute_metrics_for_pipeline(
            reviews_file=args.reviews_file,
            groups_file=cfg["groups_file"],
            personas_file=cfg["personas_file"],
            spec_file=cfg["spec_file"],
            tests_file=cfg["tests_file"],
        )

        save_json(cfg["output_file"], metrics)
        print(f"Saved {pipeline_name} metrics -> {cfg['output_file']}")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.pipeline == "all":
        summary = build_metrics_summary()
        save_json("metrics/metrics_summary.json", summary)
        print("Saved summary metrics -> metrics/metrics_summary.json")


if __name__ == "__main__":
    main()