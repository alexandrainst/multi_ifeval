"""Translate the IFEval dataset to different languages.

Usage:
    uv run src/scripts/translate_ifeval.py [--model MODEL]
"""

from pathlib import Path

import click
from datasets import disable_progress_bars
from dotenv import load_dotenv
from tqdm.auto import tqdm

from multi_ifeval.constants import LANGUAGES_COVERED_BY_LINGUA
from multi_ifeval.data_loading import (
    load_ifeval,
    load_mapping_from_language_to_example_text,
)
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

    language_to_example_text = load_mapping_from_language_to_example_text()
    languages_covered = set(LANGUAGES_COVERED_BY_LINGUA).intersection(
        set(language_to_example_text.keys())
    )

    for language in tqdm(
        iterable=languages_covered,
        desc="Translating datasets",
        total=len(language_to_example_text),
        unit="dataset",
    ):
        translated_examples = [
            translate_example(
                example=example,
                language=language,
                language_example=language_to_example_text[language],
                model=model,
            )
            for example in tqdm(
                iterable=examples,
                desc=f"Translating examples to {language.name}",
                total=len(examples),
                unit="example",
                leave=False,
            )
        ]
        language_output_path = output_dir / f"ifeval-{language.code}.jsonl"
        with language_output_path.open("w") as f:
            for example in translated_examples:
                f.write(example.model_dump_json() + "\n")


if __name__ == "__main__":
    main()
