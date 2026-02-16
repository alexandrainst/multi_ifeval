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
    # Change the instruction IDs manually
    if language not in LANGUAGES_COVERED_BY_LINGUA:
        example.instruction_id_list = [
            instruction_id.replace("english_", "")
            for instruction_id, kwargs in zip(
                example.instruction_id_list, example.kwargs
            )
            if instruction_id != "language:response_language"
        ]
    else:
        new_instruction_id_list: list[str] = list()
        new_kwargs: list[dict[str, str | int | float | bool | list[str] | None]] = []
        for instruction_id, kwargs in zip(example.instruction_id_list, example.kwargs):
            if "english_" in instruction_id:
                new_instruction_id_list.extend(
                    [
                        instruction_id.replace("english_", ""),
                        "language:response_language",
                    ]
                )
                new_kwargs.extend([kwargs, dict(language=language.code)])
            elif instruction_id == "language:response_language":
                new_instruction_id_list.append(instruction_id)
                new_kwargs.append(dict(language=language.code))
            else:
                new_instruction_id_list.append(instruction_id)
                new_kwargs.append(kwargs)

        example.instruction_id_list = new_instruction_id_list
        example.kwargs = new_kwargs

    prompt = dedent(f"""
        You are a professional translator from English to {language.name} (language
        code: {language.code!r}).

        Here is an instruction-following example in English:

        <example>
        {example.model_dump_json()}
        </example>

        You need to translate the example to {language.name}. This means the following:

        1. The `prompt` should be translated to {language.name}. You should localise
           these to {language.name} and its country's culture.
        2. The `instruction_id_list` should remain exactly the same.
        3. The keys in the kwargs dictionaries should be completely unchanged, and the
           values should _only_ be changed if they contain English words or phrases,
           which would need to be changed to the equivalent in {language.name}.

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

    def validate_generated_example(generated_example: GeneratedExample) -> None:
        """Validate the generated example.

        Args:
            generated_example:
                The generated example.

        Raises:
            ValueError:
                If the generated example is invalid.
        """
        instruction_ids_are_unchanged = (
            example.instruction_id_list == generated_example.new_instruction_id_list
        )
        if not instruction_ids_are_unchanged:
            raise ValueError(
                f"The instruction IDs were not unchanged. Expected "
                f"{example.instruction_id_list!r}, but got "
                f"{generated_example.new_instruction_id_list!r}."
            )

        for kwargs, generated_kwargs in zip(
            example.kwargs, generated_example.new_kwargs
        ):
            if list(kwargs.keys()) != [kwarg.name for kwarg in generated_kwargs]:
                raise ValueError(
                    f"The keyword argument names were not unchanged. Expected "
                    f"{list(kwargs.keys())!r}, but got "
                    f"{[kwarg.name for kwarg in generated_kwargs]!r}."
                )

    generated_example = generate(
        prompt=prompt,
        model=model,
        temperature=0.0,
        max_tokens=2048,
        response_format=GeneratedExample,
        validation_fn=validate_generated_example,
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
