"""Translation of instruction-following examples."""

from textwrap import dedent

from .data_models import Example
from .llm import generate


def translate_example(
    example: Example, language: str, language_example: str, model: str
) -> Example:
    """Translate an instruction-following example to a different language.

    Args:
        example:
            The example to translate.
        language:
            The language code to translate the example to.
        language_example:
            An example of some text written in the target language.
        model:
            The model to use for translation.

    Returns:
        The translated example.
    """
    prompt = dedent(f"""
        You are a professional translator from 'en' to {language}.

        Here is an instruction-following example in 'en':

        <example>
        {example.model_dump_json()}
        </example>

        You need to translate the example to {language!r}. This means the following:

        1. The `key` should remain the same.
        2. The `prompt` should be translated to {language!r}. If the prompt concerns
           landmarks or individuals specific to USA or to the English language, you
           should localise these to the target language and culture.
        3. The `instruction_id_list` should only be translated if the instruction IDs
           specifically mention the English language. The ID itself should remain in
           English. In almost all cases, the `instruction_id_list` should remain
           unchanged.
        4. The `kwargs` are the keyword arguments for the instruction functions related
           to the instruction IDs in the `instruction_id_list`. These should remain
           unchanged unless they contain 'en' words, in which case they should be
           translated to {language!r}.

        Here is an example of some text written in {language!r}:

        <text-written-in-target-language>
        {language_example}
        </text-written-in-target-language>

        You should return the translated example in JSON format, with the same structure
        as the original example:

        - `key` (int): The key of the example.
        - `prompt` (str): The translated prompt.
        - `instruction_id_list` (list[str]): The translated instruction ID list.
        - `kwargs` (list[dict[str, Any]]): The translated keyword arguments.
    """).strip()

    translated_example = generate(
        prompt=prompt,
        model=model,
        temperature=0.0,
        max_tokens=2048,
        response_model=Example,
    )
    assert isinstance(translated_example, Example), (
        f"Expected an Example object, but got {type(translated_example)}"
    )

    return translated_example
