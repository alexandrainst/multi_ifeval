"""Push the translated JSONL files to the Hugging Face Hub as a single dataset."""

import time
from pathlib import Path

import bits_and_bobs as bnb
import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError
from loguru import logger
from tqdm.auto import tqdm

from multi_ifeval.data_models import Example


def main() -> None:
    """Push the translated JSONL files to the Hugging Face Hub as a single dataset.

    Raises:
        HfHubHTTPError:
            If we failed to push a dataset to the Hugging Face Hub.
    """
    target_repo = "alexandrainst/multi-ifeval"

    api = HfApi()
    existing_subsets = (
        list()
        if not api.repo_exists(repo_id=target_repo, repo_type="dataset")
        else [
            config_dict["config_name"]
            for config_dict in api.repo_info(
                repo_id=target_repo, repo_type="dataset"
            ).cardData.configs
        ]
    )

    for jsonl_path in tqdm(
        iterable=list(Path("data").glob("*.jsonl")),
        desc="Creating dataset subsets",
        unit="subset",
    ):
        language_code = jsonl_path.stem.split("-")[-1]
        if language_code in existing_subsets:
            logger.info(f"{language_code} already exists. Skipping.")

        with jsonl_path.open() as f:
            examples = [
                Example.model_validate_json(line).model_dump()
                for line in f
                if line.strip()
            ]

        try:
            df = pd.DataFrame.from_records(examples, index="key")
            dataset = Dataset.from_pandas(df)
        except Exception as e:
            logger.error(f"Failed to create dataset from {jsonl_path}: {e}. Skipping.")
            continue

        for _ in range(num_attempts := 5):
            try:
                with bnb.no_terminal_output():
                    dataset.push_to_hub(
                        target_repo,
                        config_name=language_code,
                        split="test",
                        private=True,
                        commit_message=f"Add {language_code}",
                    )
                    break
            except HfHubHTTPError as e:
                if "429" in str(e):
                    logger.warning(
                        f"Rate limit exceeded for {language_code}. Waiting a minute "
                        "and retrying."
                    )
                    time.sleep(60)
                    continue
                raise e
        else:
            logger.error(
                f"Failed to push {language_code} after {num_attempts} attempts. "
                "Skipping."
            )


if __name__ == "__main__":
    main()
