import json
import os
import re
from typing import Any

from groq import Groq


# Config

INPUT_PERSONAS_FILE = "personas/personas_auto.json"
OUTPUT_SPEC_FILE = "spec/spec_auto.md"
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# Make at least 10 requirements.
# With 8 personas from your current pipeline, 2 per persona gives 16.
REQUIREMENTS_PER_PERSONA = 2


# File Helpers

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(path: str, text: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


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
                  "Do not add any explanation."
            )

    raise ValueError(f"Failed to get valid JSON after {max_retries} attempts: {last_error}")


# Prompt Builder

def build_spec_prompt(personas: list[dict]) -> str:
    persona_summaries = []

    for p in personas:
        persona_summaries.append({
            "id": p.get("id", ""),
            "name": p.get("name", ""),
            "description": p.get("description", ""),
            "derived_from_group": p.get("derived_from_group", ""),
            "goals": p.get("goals", []),
            "pain_points": p.get("pain_points", []),
            "context": p.get("context", []),
            "constraints": p.get("constraints", []),
            "evidence_reviews": p.get("evidence_reviews", []),
        })

    return f"""
You are helping with requirements engineering for the Calm app.

Task:
Generate structured software requirements from the automated personas below.

Important rules:
- Create exactly {len(personas) * REQUIREMENTS_PER_PERSONA} requirements.
- Create exactly {REQUIREMENTS_PER_PERSONA} requirements per persona.
- Use requirement IDs in this format: FR_auto_1, FR_auto_2, FR_auto_3, ...
- Each requirement must be grounded in the persona's goals, pain points, context, or constraints.
- Each requirement must reference the persona name exactly.
- Each requirement must include traceability to the persona's review group using the exact group ID.
- Requirements must be specific, testable, and free of vague language such as:
  fast, easy, better, user-friendly, seamless, intuitive, efficient.
- The description must be written as a system requirement using "The system shall ...".
- The acceptance criteria must use a Given / When / Then structure in one sentence.
- Return JSON only.

Return JSON in exactly this schema:
{{
  "requirements": [
    {{
      "requirement_id": "FR_auto_1",
      "description": "The system shall ...",
      "source_persona": "Persona Name",
      "traceability": "Derived from review group G1",
      "acceptance_criteria": "Given ..., When ..., Then ..."
    }}
  ]
}}

Personas:
{json.dumps(persona_summaries, indent=2, ensure_ascii=False)}
""".strip()


# Validation + Formatting

def normalize_requirement(item: dict, fallback_id: str) -> dict:
    requirement_id = str(item.get("requirement_id", fallback_id)).strip()
    description = str(item.get("description", "")).strip()
    source_persona = str(item.get("source_persona", "")).strip()
    traceability = str(item.get("traceability", "")).strip()
    acceptance_criteria = str(item.get("acceptance_criteria", "")).strip()

    if not requirement_id:
        requirement_id = fallback_id

    if not description.startswith("The system shall"):
        description = f"The system shall {description[:1].lower() + description[1:]}" if description else "The system shall support the identified user need."

    if not traceability.startswith("Derived from review group "):
        traceability = f"Derived from review group {traceability}".strip()

    if "Given" not in acceptance_criteria or "When" not in acceptance_criteria or "Then" not in acceptance_criteria:
        acceptance_criteria = (
            "Given the user attempts the supported action, "
            "When the system processes the request, "
            "Then the system shall satisfy the requirement successfully."
        )

    return {
        "requirement_id": requirement_id,
        "description": description,
        "source_persona": source_persona,
        "traceability": traceability,
        "acceptance_criteria": acceptance_criteria,
    }


def validate_against_personas(requirements: list[dict], personas: list[dict]) -> list[dict]:
    valid_persona_names = {p["name"] for p in personas if p.get("name")}
    valid_group_ids = {p["derived_from_group"] for p in personas if p.get("derived_from_group")}

    cleaned = []
    for i, req in enumerate(requirements, start=1):
        req = normalize_requirement(req, f"FR_auto_{i}")

        if req["source_persona"] not in valid_persona_names and personas:
            req["source_persona"] = personas[min((i - 1) // REQUIREMENTS_PER_PERSONA, len(personas) - 1)]["name"]

        matched_group = None
        for gid in valid_group_ids:
            if gid in req["traceability"]:
                matched_group = gid
                break

        if matched_group is None and personas:
            fallback_group = personas[min((i - 1) // REQUIREMENTS_PER_PERSONA, len(personas) - 1)]["derived_from_group"]
            req["traceability"] = f"Derived from review group {fallback_group}"

        cleaned.append(req)

    # Re-number cleanly
    for i, req in enumerate(cleaned, start=1):
        req["requirement_id"] = f"FR_auto_{i}"

    return cleaned


def requirements_to_markdown(requirements: list[dict]) -> str:
    blocks = []
    for req in requirements:
        block = (
            f"# Requirement ID: {req['requirement_id']}\n"
            f"- Description: [{req['description']}]\n\n"
            f"- Source Persona: [{req['source_persona']}]\n"
            f"- Traceability: [{req['traceability']}]\n"
            f"- Acceptance Criteria:[{req['acceptance_criteria']}]\n"
        )
        blocks.append(block)

    return "\n".join(blocks).strip() + "\n"


# Main

def main() -> None:
    payload = load_json(INPUT_PERSONAS_FILE)
    personas = payload.get("personas", [])

    if not personas:
        raise ValueError("No personas found in personas/personas_auto.json")

    client = groq_client()
    prompt = build_spec_prompt(personas)
    result = call_llm_json(client, prompt)

    requirements = result.get("requirements", [])
    requirements = validate_against_personas(requirements, personas)

    # Guarantee minimum of 10
    if len(requirements) < 10:
        raise ValueError(f"Expected at least 10 requirements, got {len(requirements)}")

    markdown = requirements_to_markdown(requirements)
    save_text(OUTPUT_SPEC_FILE, markdown)

    print(f"Loaded personas: {len(personas)}")
    print(f"Generated requirements: {len(requirements)}")
    print(f"Saved: {OUTPUT_SPEC_FILE}")


if __name__ == "__main__":
    main()