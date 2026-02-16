"""Loading of data to use in the project."""

from datasets import load_dataset
from huggingface_hub import DatasetInfo, HfApi

from .data_models import Example
from .languages import Language, get_all_languages


def load_ifeval() -> list[Example]:
    """Load the IFEval dataset.

    Returns:
        A list of examples.
    """
    dataset = load_dataset("google/IFEval", split="train")
    examples = [Example.model_validate(example) for example in dataset]
    for example in examples:
        example.kwargs = [
            {key: value for key, value in kwargs.items() if value is not None}
            for kwargs in example.kwargs
        ]
    return examples


def load_languages() -> list[Language]:
    """Load a list of all the languages in MultiWikiQA.

    Returns:
        A list of languages.
    """
    api = HfApi()
    repo_info = api.repo_info("alexandrainst/multi-wiki-qa", repo_type="dataset")
    assert isinstance(repo_info, DatasetInfo), (
        f"Expected a DatasetInfo object, but got {type(repo_info)}"
    )

    language_code_to_language = get_all_languages()
    languages = [
        language_code_to_language[config["config_name"]]
        for config in repo_info.cardData.configs
        if config["config_name"] in language_code_to_language
    ]

    return languages
