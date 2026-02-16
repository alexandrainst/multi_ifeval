"""Translation of instruction-following examples."""

from textwrap import dedent

from .constants import LANGUAGES_COVERED_BY_LINGUA
from .data_models import Example, GeneratedExample
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
        f"""
        4. Note that the {language.name} language detection is not covered by language
           detection software, so if one of the instruction IDs mentions language
           detection in the source language (English), you have to remove that
           instruction ID from the output instruction_id_list and remove mention of that
           restriction from the output prompt."""
        if language not in LANGUAGES_COVERED_BY_LINGUA
        else ""
    )

    prompt = dedent(f"""
        You are a professional translator from English to {language.name} (language
        code: {language.code!r}).

        Here is an instruction-following example in English:

        <example>
        {example.model_dump_json()}
        </example>

        You need to translate the example to {language.name}. This means the following:

        1. The `prompt` should be translated to {language.name}. If the prompt concerns
           landmarks or individuals specific to USA or to the English language, you
           should localise these to the target language and culture.
        2. The `instruction_id_list` should only be translated if the instruction IDs
           specifically mention the English language. The ID itself should remain in
           English. In almost all cases, the `instruction_id_list` should remain
           unchanged.
        3. The `kwargs` are the keyword arguments for the instruction functions related
           to the instruction IDs in the `instruction_id_list`. These should remain
           unchanged unless they contain English words, in which case they should be
           translated to {language.name}.{translation_condition}

        Here is an example of some text written in {language.name}:

        <{language.code}-example>
        {language_example.replace("\n", " ")}
        </{language.code}-example>

        You should return the translated example in JSON format, with the following
        structure:

        - `new_prompt` (str): The translated prompt.
        - `new_instruction_id_list` (list[str]): The translated instruction ID list.
        - `new_kwargs` (list[list[dict]]): The translated keyword arguments.

        Here the `new_kwargs` must be a list of the same length as the
        `new_instruction_id_list`, containing lists of dicts, each with keys `name` and
        `value`, where `name` is the name of the keyword argument and `value` is the
        value of the keyword argument. Each list of kwarg dicts should be the keyword
        arguments for the corresponding instruction ID in the `new_instruction_id_list`.
    """).strip()

    generated_example = generate(
        prompt=prompt,
        model=model,
        temperature=0.0,
        max_tokens=2048,
        response_format=GeneratedExample,
    )
    return Example(
        key=example.key,
        prompt=generated_example.new_prompt,
        instruction_id_list=generated_example.new_instruction_id_list,
        kwargs=[
            {kwarg.name: kwarg.value for kwarg in kwarg_list}
            for kwarg_list in generated_example.new_kwargs
        ],
    )
