"""Push the translated JSONL files to the Hugging Face Hub as a single dataset."""

from pathlib import Path

import bits_and_bobs as bnb
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm

from multi_ifeval.data_models import Example


def main() -> None:
    """Push the translated JSONL files to the Hugging Face Hub as a single dataset."""
    target_repo = "alexandrainst/multi-ifeval"

    for jsonl_path in tqdm(
        iterable=list(Path("data").glob("*.jsonl")),
        desc="Creating dataset subsets",
        unit="subset",
    ):
        with jsonl_path.open() as f:
            examples = [
                Example.model_validate_json(line).model_dump()
                for line in f
                if line.strip()
            ]

        df = pd.DataFrame.from_records(examples, index="key")
        dataset = Dataset.from_pandas(df)

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
