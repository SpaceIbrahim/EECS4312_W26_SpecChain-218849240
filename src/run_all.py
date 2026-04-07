from pathlib import Path
import subprocess
import sys


def repo_root() -> Path:
    # go up one level from src/ to get the repo root
    return Path(__file__).resolve().parent.parent


def run_step(script_name: str, expected_outputs: list[str] | None = None) -> None:
    root = repo_root()
    script_path = root / "src" / script_name

    if not script_path.is_file():
        raise FileNotFoundError(f"script not found: {script_path}")

    print(f"\nrunning {script_name}...")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=root,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")

    print(f"{script_name} done.")

    # check that the expected output files actually got created
    if expected_outputs:
        for rel_path in expected_outputs:
            output_path = root / rel_path
            if output_path.exists():
                print(f"  found: {rel_path}")
            else:
                raise FileNotFoundError(
                    f"{script_name} finished but {rel_path} was not created"
                )


def main() -> int:
    print("starting automated SpecChain pipeline...")

    try:
        # step 1: collect raw reviews from the app store
        run_step(
            "01_collect_or_import.py",
            expected_outputs=[
                "data/reviews_raw.jsonl",
            ],
        )

        # step 2: clean and preprocess the raw reviews
        run_step(
            "02_clean.py",
            expected_outputs=[
                "data/reviews_clean.jsonl",
                "data/dataset_metadata.json",
            ],
        )

        # step 3: auto group reviews and generate personas using the LLM
        run_step(
            "05_personas_auto.py",
            expected_outputs=[
                "data/review_groups_auto.json",
                "personas/personas_auto.json",
            ],
        )

        # step 4: generate the spec from the auto personas
        run_step(
            "06_spec_generate.py",
            expected_outputs=[
                "spec/spec_auto.md",
            ],
        )

        # step 5: generate validation tests from the spec
        run_step(
            "07_tests_generate.py",
            expected_outputs=[
                "tests/tests_auto.json",
            ],
        )

        # step 6: compute metrics for all three pipelines and save summary
        run_step(
            "08_metrics.py",
            expected_outputs=[
                "metrics/metrics_auto.json",
                "metrics/metrics_summary.json",
            ],
        )

        print("\npipeline finished successfully, all steps completed without errors.")
        return 0

    except Exception as exc:
        print(f"\nERROR: {exc}")
        print("pipeline stopped.")
        return 1


if __name__ == "__main__":
    sys.exit(main())