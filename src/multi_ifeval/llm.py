"""Generation with a large language model."""

import typing as t

import litellm
from litellm import Choices, ModelResponse

T = t.TypeVar("T")


def generate(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    response_model: type[T] | None = None,
) -> str | T:
    """Generate a response to a prompt.

    Args:
        prompt:
            The prompt to generate a response to.
        model:
            The model to use for generation.
        temperature:
            The temperature to use for generation.
        max_tokens:
            The maximum number of tokens to generate.
        response_model (optional):
            The model to use for generation. If None then the response is returned as a
            string. Defaults to None.

    Returns:
        The generated response, which is a Pydantic model if `response_model` is set,
        and otherwise a string.
    """
    response: ModelResponse = litellm.completion(  # pyrefly: ignore[not-callable]
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        response_model=response_model,
    )
    choice = response.choices[0]
    assert isinstance(choice, Choices), (
        f"Expected a Choices object, but got {type(choice)}"
    )
    completion = choice.message.content
    assert completion is not None, f"The model did not return a completion: {response}"

    if response_model is not None:
        return response_model.model_validate_json(completion)
    return completion
