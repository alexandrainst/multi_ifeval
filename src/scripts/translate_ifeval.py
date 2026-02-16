"""Translate the IFEval dataset to different languages.

Usage:
    uv run src/scripts/translate_ifeval.py [--model MODEL]
"""

import warnings
from pathlib import Path
from string import punctuation

import click
from datasets import Dataset, DownloadConfig, disable_progress_bars, load_dataset
from dotenv import load_dotenv
from tqdm.auto import tqdm

from multi_ifeval.data_loading import load_ifeval, load_languages
from multi_ifeval.translation import translate_example

load_dotenv()


@click.command()
@click.option(
    "--model",
    "-m",
    type=str,
    default="gemini/gemini-3-flash-preview",
    help="The model to use for translation.",
)
def main(model: str) -> None:
    """Translate the IFEval dataset to different languages."""
    disable_progress_bars()

    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    examples = load_ifeval()

    for language in tqdm(
        iterable=load_languages(), desc="Translating datasets", unit="dataset"
    ):
        language_output_path = output_dir / f"ifeval-{language.code}.jsonl"
        if language_output_path.exists():
            continue

        dataset = load_dataset(
            "alexandrainst/multi-wiki-qa",
            name=language.code,
            split="train",
            download_config=DownloadConfig(disable_tqdm=True),
        )

        for example in tqdm(
            iterable=examples,
            desc=f"Translating examples to {language.name}",
            total=len(examples),
            unit="example",
            leave=False,
        ):
            # Load the example text
            example_text = dataset.shuffle()[0]["context"]
            special_symbol_fraction = sum(
                1 for char in example_text if char in punctuation
            ) / len(example_text)
            while special_symbol_fraction > 0.05:
                example_text = dataset.shuffle()[0]["context"]
                special_symbol_fraction = sum(
                    1 for char in example_text if char in punctuation
                ) / len(example_text)
            assert isinstance(example_text, str), (
                f"Expected a string, but got {type(example_text)}"
            )

            # Remove the example text from the dataset
            dataset = dataset.filter(lambda x: x["context"] != example_text)
            assert isinstance(dataset, Dataset), (
                f"Expected a Dataset, but got {type(dataset)}"
            )

            error_msgs: list[str] = list()
            for _ in range(num_attempts := 3):
                try:
                    translated_example = translate_example(
                        example=example,
                        language=language,
                        language_example=example_text,
                        model=model,
                    )
                    break
                except Exception as e:
                    error_msgs.append(str(e))
            else:
                warnings.warn(
                    f"Failed to translate example {example.key} to {language.name}, "
                    f"after {num_attempts} attempts. Skipping. Here are the errors "
                    f"that occurred:\n{error_msgs}"
                )
                continue

            with language_output_path.open("a") as f:
                f.write(translated_example.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
