"""Loading of data to use in the project."""

from datasets import load_dataset
from huggingface_hub import DatasetInfo, HfApi

from .data_models import Example


def load_ifeval() -> list[Example]:
    """Load the IFEval dataset.

    Returns:
        A list of examples.
    """
    dataset = load_dataset("google/IFEval", split="train")
    examples = [Example.model_validate(example) for example in dataset]
    return examples


def load_mapping_from_language_to_example_text() -> dict[str, str]:
    """Load a sample text for each language from the MultiWikiQA dataset.

    Returns:
        A dictionary mapping language codes to a text example written in that language.
    """
    # Get the list of language codes in MultiWikiQA
    api = HfApi()
    repo_info = api.repo_info("alexandrainst/multi-wiki-qa", repo_type="dataset")
    assert isinstance(repo_info, DatasetInfo), (
        f"Expected a DatasetInfo object, but got {type(repo_info)}"
    )
    languages = repo_info.cardData.config_names

    language_to_example_text: dict[str, str] = dict()
    for language in languages:
        dataset = load_dataset(
            "alexandrainst/multi-wiki-qa", name=language, split="train"
        )
        example = dataset.shuffle(seed=42)[0]["context"]
        assert isinstance(example, str), f"Expected a string, but got {type(example)}"
        language_to_example_text[language] = example

    return language_to_example_text
