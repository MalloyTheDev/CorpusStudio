from pathlib import Path
import subprocess
import sys


def main() -> int:
    examples = [
        ("examples/datasets/raw_text/train.jsonl", "raw_text"),
        ("examples/datasets/instruction/train.jsonl", "instruction"),
        ("examples/datasets/chat/train.jsonl", "chat"),
        ("examples/datasets/preference/train.jsonl", "preference"),
        ("examples/datasets/code/train.jsonl", "code"),
        ("examples/datasets/image_caption/train.jsonl", "image_caption"),
        ("examples/datasets/retrieval/train.jsonl", "retrieval"),
        ("examples/datasets/evaluation/test.jsonl", "evaluation"),
    ]

    engine_dir = Path(__file__).resolve().parents[1] / "engine"

    for rel_path, schema in examples:
        path = Path(__file__).resolve().parents[1] / rel_path
        print(f"Validating {rel_path} as {schema}")
        result = subprocess.run(
            [sys.executable, "-m", "corpus_studio.cli", "validate", str(path), schema],
            cwd=engine_dir,
            text=True,
        )
        if result.returncode != 0:
            return result.returncode

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
