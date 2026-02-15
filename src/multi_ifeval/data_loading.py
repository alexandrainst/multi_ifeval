"""Loading of data to use in the project."""

import json
from pathlib import Path
from string import punctuation

from datasets import IterableDataset, load_dataset
from huggingface_hub import DatasetInfo, HfApi
from tqdm.auto import tqdm

from .data_models import Example
from .languages import Language, get_all_languages


def load_ifeval() -> list[Example]:
    """Load the IFEval dataset.

    Returns:
        A list of examples.
    """
    dataset = load_dataset("google/IFEval", split="train")
    examples = [Example.model_validate(example) for example in dataset]
    return examples


def load_mapping_from_language_to_example_text() -> dict[Language, str]:
    """Load a sample text for each language from the MultiWikiQA dataset.

    Returns:
        A dictionary mapping language codes to a text example written in that language.
    """
    mapping_path = Path("data", "mapping_from_language_to_example_text.json")
    mapping_path.parent.mkdir(exist_ok=True, parents=True)

    language_code_to_language = get_all_languages()

    # Get the list of language codes in MultiWikiQA
    api = HfApi()
    repo_info = api.repo_info("alexandrainst/multi-wiki-qa", repo_type="dataset")
    assert isinstance(repo_info, DatasetInfo), (
        f"Expected a DatasetInfo object, but got {type(repo_info)}"
    )
    languages = [
        language_code_to_language[config["config_name"]]
        for config in repo_info.cardData.configs
        if config["config_name"] in language_code_to_language
    ]

    language_code_to_example_text: dict[str, str] = dict()
    if mapping_path.exists():
        with mapping_path.open() as file:
            language_code_to_example_text = json.load(file)

    for language in tqdm(
        iterable=languages,
        desc="Loading sample text for each language",
        unit="language",
    ):
        if language.code in language_code_to_example_text:
            continue

        dataset = load_dataset(
            "alexandrainst/multi-wiki-qa",
            name=language.code,
            split="train",
            streaming=True,
        )
        assert isinstance(dataset, IterableDataset), (
            f"Expected an IterableDataset object, but got {type(dataset)}"
        )

        # Keep trying to fetch examples until we find an example which is not full of
        # special symbols, such as tables
        itr = iter(dataset)
        example_text: str = next(itr)["context"]
        special_symbol_fraction = sum(
            1 for char in example_text if char in punctuation
        ) / len(example_text)
        while special_symbol_fraction > 0.05:
            example_text = next(itr)["context"]
            special_symbol_fraction = sum(
                1 for char in example_text if char in punctuation
            ) / len(example_text)

        language_code_to_example_text[language.code] = example_text
        with mapping_path.open("w") as file:
            json.dump(language_code_to_example_text, file, indent=2, ensure_ascii=False)

    return {
        language: language_code_to_example_text[language.code] for language in languages
    }
