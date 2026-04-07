import json
import os
import re
from typing import Any

from groq import Groq


# Config

INPUT_SPEC_FILE = "spec/spec_auto.md"
OUTPUT_TESTS_FILE = "tests/tests_auto.json"
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# You can raise this to 2 later if you want stronger coverage.
TESTS_PER_REQUIREMENT = 1


# File Helpers

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_json(path: str, data: Any) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# LLM Helpers

def groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY is not set in the environment.")
    return Groq(api_key=api_key)


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
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

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
    return json.loads(candidate)


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
            prompt = (
                prompt
                + "\n\nIMPORTANT: Your previous answer was invalid. "
                  "Return only one valid JSON object. "
                  "Do not use markdown. "
                  "Do not add explanation."
            )

    raise ValueError(f"Failed to get valid JSON after {max_retries} attempts: {last_error}")


# Spec Parsing

def parse_spec_markdown(markdown_text: str) -> list[dict]:
    """
    Parses requirement blocks in this format:

    # Requirement ID: FR_auto_1
    - Description: [The system shall ...]
    - Source Persona: [Persona Name]
    - Traceability: [Derived from review group G1]
    - Acceptance Criteria:[Given ..., When ..., Then ...]

    Returns a list of requirement dictionaries.
    """
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


# Prompt Builder

def build_test_prompt(requirements: list[dict]) -> str:
    prompt_requirements = []
    for req in requirements:
        prompt_requirements.append({
            "requirement_id": req["requirement_id"],
            "description": req["description"],
            "source_persona": req["source_persona"],
            "traceability": req["traceability"],
            "acceptance_criteria": req["acceptance_criteria"],
        })

    total_tests = len(requirements) * TESTS_PER_REQUIREMENT

    return f"""
You are helping with requirements engineering for the Calm app.

Task:
Generate validation tests from the software requirements below.

Rules:
- Create exactly {TESTS_PER_REQUIREMENT} test(s) per requirement.
- Create exactly {total_tests} tests total.
- Each test must reference exactly one requirement_id.
- Every requirement must have at least one associated test.
- Use test IDs in this format: T_auto_1, T_auto_2, T_auto_3, ...
- scenario must be short and specific.
- steps must be a clear ordered list of user/system actions.
- expected_result must clearly reflect the linked requirement.
- Keep all tests grounded in the requirement text.
- Return JSON only.

Return JSON in exactly this schema:
{{
  "tests": [
    {{
      "test_id": "T_auto_1",
      "requirement_id": "FR_auto_1",
      "scenario": "Short scenario name",
      "steps": [
        "Step 1",
        "Step 2",
        "Step 3"
      ],
      "expected_result": "Expected outcome here."
    }}
  ]
}}

Requirements:
{json.dumps(prompt_requirements, indent=2, ensure_ascii=False)}
""".strip()


# Validation + Normalization

def normalize_test(test: dict, fallback_id: str) -> dict:
    test_id = str(test.get("test_id", fallback_id)).strip() or fallback_id
    requirement_id = str(test.get("requirement_id", "")).strip()
    scenario = str(test.get("scenario", "")).strip()
    expected_result = str(test.get("expected_result", "")).strip()

    raw_steps = test.get("steps", [])
    if not isinstance(raw_steps, list):
        raw_steps = []

    steps = [str(step).strip() for step in raw_steps if str(step).strip()]

    if not scenario:
        scenario = "Requirement validation scenario"

    if len(steps) < 3:
        steps = [
            "Open the relevant Calm feature.",
            "Perform the action described in the requirement.",
            "Observe the system response."
        ]

    if not expected_result:
        expected_result = "The system satisfies the requirement successfully."

    return {
        "test_id": test_id,
        "requirement_id": requirement_id,
        "scenario": scenario,
        "steps": steps,
        "expected_result": expected_result,
    }


def validate_tests_against_requirements(tests: list[dict], requirements: list[dict]) -> list[dict]:
    requirement_ids = [r["requirement_id"] for r in requirements]
    requirement_set = set(requirement_ids)

    normalized = []
    for i, test in enumerate(tests, start=1):
        item = normalize_test(test, f"T_auto_{i}")

        if item["requirement_id"] not in requirement_set and requirements:
            fallback_req = requirements[min((i - 1) // TESTS_PER_REQUIREMENT, len(requirements) - 1)]
            item["requirement_id"] = fallback_req["requirement_id"]

        normalized.append(item)

    # Renumber test IDs cleanly
    for i, item in enumerate(normalized, start=1):
        item["test_id"] = f"T_auto_{i}"

    # Ensure every requirement has at least one test
    covered = {t["requirement_id"] for t in normalized}
    missing = [rid for rid in requirement_ids if rid not in covered]

    for rid in missing:
        req = next(r for r in requirements if r["requirement_id"] == rid)
        normalized.append({
            "test_id": f"T_auto_{len(normalized) + 1}",
            "requirement_id": rid,
            "scenario": f"Validation of {rid}",
            "steps": [
                "Open the relevant Calm feature.",
                f"Perform the behavior described in requirement {rid}.",
                "Observe the system outcome."
            ],
            "expected_result": req["description"]
        })

    return normalized


# Main

def main() -> None:
    spec_text = load_text(INPUT_SPEC_FILE)
    requirements = parse_spec_markdown(spec_text)

    if not requirements:
        raise ValueError(
            "No requirements were parsed from spec/spec_auto.md. "
            "Check that the markdown format matches the expected template."
        )

    print(f"Parsed requirements: {len(requirements)}")

    client = groq_client()
    prompt = build_test_prompt(requirements)
    result = call_llm_json(client, prompt)

    tests = result.get("tests", [])
    tests = validate_tests_against_requirements(tests, requirements)

    payload = {"tests": tests}
    save_json(OUTPUT_TESTS_FILE, payload)

    print(f"Generated tests: {len(tests)}")
    print(f"Saved: {OUTPUT_TESTS_FILE}")


if __name__ == "__main__":
    main()