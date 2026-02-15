"""Translation of instruction-following examples."""

from textwrap import dedent

from .constants import LANGUAGES_COVERED_BY_LINGUA
from .data_models import Example
from .languages import Language
from .llm import generate


def translate_example(
    example: Example, language: Language, language_example: str, model: str
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
    translation_condition = (
        (
            "\n"
            "5. Note that the {language.name} language detection is not covered by "
            "   language detection software, so if one of the instruction IDs mentions "
            "   language detection in the source language (English), you have to "
            "   remove that instruction ID from the output instruction_id_list and "
            "   remove mention of that restriction from the output prompt."
        )
        if language not in LANGUAGES_COVERED_BY_LINGUA
        else ""
    )

    prompt = dedent(f"""
        You are a professional translator from English to {language.name} (language
        code: {language.code!r}).

        Here is an instruction-following example in 'en':

        <example>
        {example.model_dump_json()}
        </example>

        You need to translate the example to {language.name}. This means the following:

        1. The `key` should remain the same.
        2. The `prompt` should be translated to {language.name}. If the prompt concerns
           landmarks or individuals specific to USA or to the English language, you
           should localise these to the target language and culture.
        3. The `instruction_id_list` should only be translated if the instruction IDs
           specifically mention the English language. The ID itself should remain in
           English. In almost all cases, the `instruction_id_list` should remain
           unchanged.
        4. The `kwargs` are the keyword arguments for the instruction functions related
           to the instruction IDs in the `instruction_id_list`. These should remain
           unchanged unless they contain 'en' words, in which case they should be
           translated to {language.name}.{translation_condition}

        Here is an example of some text written in {language.name}:

        <{language.code}-example>
        {language_example}
        </{language.code}-example>

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
