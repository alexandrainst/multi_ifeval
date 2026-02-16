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

            # Ensure that the example text is not full of special symbols like tables
            best_example_text = example_text
            best_special_symbol_fraction = special_symbol_fraction
            for _ in range(10):
                if special_symbol_fraction < 0.05:
                    break
                example_text = dataset.shuffle()[0]["context"]
                special_symbol_fraction = sum(
                    1 for char in example_text if char in punctuation
                ) / len(example_text)
                if special_symbol_fraction < best_special_symbol_fraction:
                    best_special_symbol_fraction = special_symbol_fraction
                    best_example_text = example_text
            else:
                example_text = best_example_text

            assert isinstance(example_text, str), (
                f"Expected a string, but got {type(example_text)}"
            )

            # Remove the example text from the dataset, unless it's the last example
            filtered_dataset = dataset.filter(lambda x: x["context"] != example_text)
            if len(filtered_dataset) > 1:
                dataset = filtered_dataset
                assert isinstance(dataset, Dataset), (
                    f"Expected a Dataset, but got {type(dataset)}"
                )

            try:
                translated_example = translate_example(
                    example=example,
                    language=language,
                    language_example=example_text,
                    model=model,
                )
            except Exception as e:
                warnings.warn(
                    f"Failed to translate example {example.key} to {language.name}. "
                    f"Skipping. Here are the errors that occurred:\n{e}"
                )
                continue

            with language_output_path.open("a") as f:
                f.write(translated_example.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
