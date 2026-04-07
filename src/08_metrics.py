import argparse
import json
import os
import re
from typing import Any


# file paths for each pipeline
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

# the summary json uses "automated" instead of "auto"
SUMMARY_KEY_MAP = {
    "manual": "manual",
    "auto": "automated",
    "hybrid": "hybrid",
}

# words that make a requirement vague/untestable
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

SUMMARY_OUTPUT_FILE = "metrics/metrics_summary.json"


# helper to create folders if they dont exist before saving
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


# parse the spec markdown file and pull out each requirement block
def parse_spec_markdown(markdown_text: str, source_path: str = "") -> list[dict]:
    # try matching fields wrapped in square brackets first
    pattern = re.compile(
        r"#+\s*Requirement\s+ID:\s*(?P<requirement_id>[^\n]+)\n"
        r"[\s\S]*?-\s*Description:\s*\[(?P<description>.*?)\]\s*\n+"
        r"[\s\S]*?-\s*Source\s*Persona:\s*\[(?P<source_persona>.*?)\]\s*\n"
        r"[\s\S]*?-\s*Traceability:\s*\[(?P<traceability>.*?)\]\s*\n"
        r"[\s\S]*?-\s*Acceptance\s*Criteria:\s*\[(?P<acceptance_criteria>.*?)\]",
        re.DOTALL | re.IGNORECASE,
    )

    # fallback in case the spec doesnt use brackets
    fallback_pattern = re.compile(
        r"#+\s*Requirement\s+ID:\s*(?P<requirement_id>[^\n]+)\n"
        r"[\s\S]*?-\s*Description:\s*(?P<description>[^\n\[]+)\n"
        r"[\s\S]*?-\s*Source\s*Persona:\s*(?P<source_persona>[^\n\[]+)\n"
        r"[\s\S]*?-\s*Traceability:\s*(?P<traceability>[^\n\[]+)\n"
        r"[\s\S]*?-\s*Acceptance\s*Criteria:\s*(?P<acceptance_criteria>[^\n\[]+)",
        re.DOTALL | re.IGNORECASE,
    )

    requirements = []
    for match in pattern.finditer(markdown_text):
        requirements.append(
            {
                "requirement_id": match.group("requirement_id").strip(),
                "description": " ".join(match.group("description").split()),
                "source_persona": " ".join(match.group("source_persona").split()),
                "traceability": " ".join(match.group("traceability").split()),
                "acceptance_criteria": " ".join(
                    match.group("acceptance_criteria").split()
                ),
            }
        )

    # if the first pattern found nothing, try the fallback
    if not requirements:
        for match in fallback_pattern.finditer(markdown_text):
            requirements.append(
                {
                    "requirement_id": match.group("requirement_id").strip(),
                    "description": " ".join(match.group("description").split()),
                    "source_persona": " ".join(match.group("source_persona").split()),
                    "traceability": " ".join(match.group("traceability").split()),
                    "acceptance_criteria": " ".join(
                        match.group("acceptance_criteria").split()
                    ),
                }
            )

    if not requirements:
        print(f"  [WARNING] could not parse any requirements from '{source_path}'")
        print(f"  make sure headings say '# Requirement ID:' and fields use")
        print(f"  '- Description:', '- Source Persona:', '- Traceability:', '- Acceptance Criteria:'")

    return requirements


def load_requirements(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return parse_spec_markdown(text, source_path=path)


def round_ratio(value: float) -> float:
    return round(value, 2)


# the groups/personas files use different key names depending on the pipeline
# so we check a few common ones
GROUP_INDEX_FIELDS = [
    "review_indexes",
    "review_indices",
    "review_ids",
    "reviews",
    "review_index",
]

PERSONA_REVIEW_FIELDS = [
    "evidence_reviews",
    "review_indexes",
    "review_indices",
    "review_ids",
    "reviews",
]


def _get_review_list(obj: dict, field_names: list[str]) -> list:
    # return the first field that actually has data
    for field in field_names:
        value = obj.get(field)
        if isinstance(value, list) and value:
            return value
    return []


def collect_review_indexes_from_groups(groups: list[dict]) -> set[str]:
    covered = set()
    for group in groups:
        for rid in _get_review_list(group, GROUP_INDEX_FIELDS):
            covered.add(str(rid))
    if not covered and groups:
        found_keys = {k for g in groups for k in g.keys()}
        print(f"  [WARNING] no review IDs found in groups. keys seen: {found_keys}")
    return covered


def collect_review_indexes_from_personas(personas: list[dict]) -> set[str]:
    covered = set()
    for persona in personas:
        for rid in _get_review_list(persona, PERSONA_REVIEW_FIELDS):
            covered.add(str(rid))
    if not covered and personas:
        found_keys = {k for p in personas for k in p.keys()}
        print(f"  [WARNING] no review IDs found in personas. keys seen: {found_keys}")
    return covered


def requirement_is_traceable(req: dict) -> bool:
    # a requirement is traceable if it has both a persona and a traceability field
    has_persona = bool(req.get("source_persona", "").strip())
    has_traceability = bool(req.get("traceability", "").strip())
    return has_persona and has_traceability


def requirement_is_ambiguous(req: dict) -> bool:
    # combine description and acceptance criteria then check for vague words
    text = f"{req.get('description', '')} {req.get('acceptance_criteria', '')}".lower()
    normalized = re.sub(r"[^a-z0-9\s\-]", " ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    for term in AMBIGUOUS_TERMS:
        if term in normalized:
            return True
    return False


def compute_traceability_links(
    personas: list[dict], requirements: list[dict], tests: list[dict]
) -> int:
    # count links across all three layers: group->persona, persona->req, req->test
    persona_to_group = sum(1 for p in personas if p.get("derived_from_group"))
    req_to_persona = sum(1 for req in requirements if requirement_is_traceable(req))
    test_to_req = sum(1 for t in tests if str(t.get("requirement_id", "")).strip())
    return persona_to_group + req_to_persona + test_to_req


def compute_metrics_for_pipeline(
    pipeline_name: str,
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

    # prefer group-level coverage, fall back to persona evidence if groups are empty
    covered_reviews = collect_review_indexes_from_groups(groups)
    if not covered_reviews:
        covered_reviews = collect_review_indexes_from_personas(personas)

    review_coverage_ratio = (
        round_ratio(len(covered_reviews) / dataset_size) if dataset_size > 0 else 0.0
    )

    traceable_count = sum(1 for req in requirements if requirement_is_traceable(req))
    traceability_ratio = (
        round_ratio(traceable_count / requirements_count) if requirements_count > 0 else 0.0
    )

    # get the set of requirement IDs that have at least one test
    tested_ids = {
        str(t.get("requirement_id", "")).strip()
        for t in tests
        if str(t.get("requirement_id", "")).strip()
    }
    testable_count = sum(1 for req in requirements if req["requirement_id"] in tested_ids)
    testability_rate = (
        round_ratio(testable_count / requirements_count) if requirements_count > 0 else 0.0
    )

    ambiguous_count = sum(1 for req in requirements if requirement_is_ambiguous(req))
    ambiguity_ratio = (
        round_ratio(ambiguous_count / requirements_count) if requirements_count > 0 else 0.0
    )

    traceability_links = compute_traceability_links(personas, requirements, tests)

    return {
        "pipeline": pipeline_name,
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


def build_metrics_summary(computed: dict[str, dict]) -> dict:
    # build the summary json, renaming "auto" to "automated" to match the expected format
    # also strip the "pipeline" field since it's redundant in the summary
    summary = {}
    for pipeline_name, metrics in computed.items():
        summary_key = SUMMARY_KEY_MAP.get(pipeline_name, pipeline_name)
        entry = {k: v for k, v in metrics.items() if k != "pipeline"}
        summary[summary_key] = entry
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute metrics for the manual, auto, and hybrid pipelines."
    )
    parser.add_argument(
        "--pipeline",
        choices=["manual", "auto", "hybrid", "all"],
        default="all",
        help="which pipeline to compute metrics for (default: all)",
    )
    parser.add_argument(
        "--reviews-file",
        default=DEFAULT_REVIEWS_FILE,
        help="path to the cleaned reviews jsonl file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipelines = (
        ["manual", "auto", "hybrid"] if args.pipeline == "all" else [args.pipeline]
    )

    computed: dict[str, dict] = {}

    for pipeline_name in pipelines:
        cfg = PIPELINE_CONFIGS[pipeline_name]

        print(f"\ncomputing metrics for: {pipeline_name}")
        metrics = compute_metrics_for_pipeline(
            pipeline_name=pipeline_name,
            reviews_file=args.reviews_file,
            groups_file=cfg["groups_file"],
            personas_file=cfg["personas_file"],
            spec_file=cfg["spec_file"],
            tests_file=cfg["tests_file"],
        )

        save_json(cfg["output_file"], metrics)
        print(f"saved -> {cfg['output_file']}")
        print(json.dumps(metrics, indent=2, ensure_ascii=False))

        computed[pipeline_name] = metrics

    # write the combined summary file when all three are computed
    if args.pipeline == "all":
        summary = build_metrics_summary(computed)
        save_json(SUMMARY_OUTPUT_FILE, summary)
        print(f"\nSaved summary -> {SUMMARY_OUTPUT_FILE}")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()