"""Push the translated JSONL files to the Hugging Face Hub as a single dataset."""

import json
from pathlib import Path

import bits_and_bobs as bnb
from datasets import Dataset
from tqdm.auto import tqdm


def main() -> None:
    """Push the translated JSONL files to the Hugging Face Hub as a single dataset."""
    target_repo = "alexandrainst/multi-ifeval"

    for jsonl_path in tqdm(
        iterable=list(Path("data").glob("*.jsonl")),
        desc="Creating dataset subsets",
        unit="subset",
    ):
        with jsonl_path.open() as f:
            examples = [json.loads(line) for line in f if line.strip()]
        dataset = Dataset.from_list(examples)

        language_code = jsonl_path.stem.split("-")[-1]
        with bnb.no_terminal_output():
            dataset.push_to_hub(
                target_repo,
                config_name=language_code,
                split="test",
                private=True,
                commit_message=f"Add {language_code}",
            )


if __name__ == "__main__":
    main()
