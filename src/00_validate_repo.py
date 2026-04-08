from pathlib import Path
import sys
import importlib.util

# checks that all required folders, files, and libraries are present
# run from the repo root python src/00_validate_repo.py


def get_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    repo_root = get_repo_root()

    required_dirs = [
        "data", "personas", "spec", "metrics", "reflection", "tests", "src",
    ]

    required_files = [
        "README.md",
        "data/reviews_raw.jsonl",
        "data/reviews_clean.jsonl",
        "data/dataset_metadata.json",
        "data/review_groups_manual.json",
        "data/review_groups_auto.json",
        "data/review_groups_hybrid.json",
        "personas/personas_manual.json",
        "personas/personas_auto.json",
        "personas/personas_hybrid.json",
        "spec/spec_manual.md",
        "spec/spec_auto.md",
        "spec/spec_hybrid.md",
        "metrics/metrics_manual.json",
        "metrics/metrics_auto.json",
        "metrics/metrics_hybrid.json",
        "metrics/metrics_summary.json",
        "reflection/reflection.md",
        "tests/tests_manual.json",
        "tests/tests_auto.json",
        "tests/tests_hybrid.json",
        "src/00_validate_repo.py",
        "src/01_collect_or_import.py",
        "src/02_clean.py",
        "src/03_manual_coding_template.py",
        "src/04_personas_manual.py",
        "src/05_personas_auto.py",
        "src/06_spec_generate.py",
        "src/07_tests_generate.py",
        "src/08_metrics.py",
        "src/run_all.py",
    ]

    required_libraries = [
        "groq", "sklearn", "numpy", "nltk", "google_play_scraper", "num2words", "sentence-transformers",
    ]

    missing_dirs = []
    missing_files = []
    missing_libraries = []

    print("checking repository structure...\n")

    for folder in required_dirs:
        if (repo_root / folder).is_dir():
            print(f"  {folder}/ found")
        else:
            print(f"  {folder}/ MISSING")
            missing_dirs.append(folder)

    print()

    for f in required_files:
        if (repo_root / f).is_file():
            print(f"  {f} found")
        else:
            print(f"  {f} MISSING")
            missing_files.append(f)

    print("\nchecking python libraries...\n")

    for lib in required_libraries:
        if module_exists(lib):
            print(f"  {lib} found")
        else:
            print(f"  {lib} MISSING")
            missing_libraries.append(lib)

    print("\nvalidation complete")

    if not missing_dirs and not missing_files and not missing_libraries:
        print("everything looks good, all folders, files, and libraries are present.")
        return 0

    # print a summary of what's missing
    print("some items are missing:\n")
    for folder in missing_dirs:
        print(f"  missing folder: {folder}/")
    for f in missing_files:
        print(f"  missing file: {f}")
    for lib in missing_libraries:
        print(f"  missing library: {lib}")

    return 1


if __name__ == "__main__":
    sys.exit(main())