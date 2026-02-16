"""Generation with a large language model."""

import collections.abc as c
import typing as t

import litellm
from litellm import Choices, ModelResponse

T = t.TypeVar("T")


def generate(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    response_format: type[T] | None = None,
    validation_fn: c.Callable[[T], T] = lambda x: x,
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
        response_format (optional):
            The model to use for generation. If None then the response is returned as a
            string. Defaults to None.
        validation_fn (optional):
            A function that validates the response. Defaults to no validation.

    Returns:
        The generated response, which is a Pydantic model if `response_format` is set,
        and otherwise a string.

    Raises:
        RuntimeError:
            If the model failed to generate a response that fit the response format.
            This is only relevant if there is some post-initialisation validation
            happening in the response format.
    """
    conversation = [dict(role="user", content=prompt)]
    response: ModelResponse = litellm.completion(  # pyrefly: ignore[not-callable]
        model=model,
        messages=conversation,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
    )
    choice = response.choices[0]
    assert isinstance(choice, Choices), (
        f"Expected a Choices object, but got {type(choice)}"
    )
    completion = choice.message.content
    assert completion is not None, f"The model did not return a completion: {response}"

    error_msgs: list[str] = list()
    if response_format is not None:
        for _ in range(num_attempts := 3):
            try:
                output = response_format.model_validate_json(completion)
                output = validation_fn(output)
                return output

            except Exception as e:
                error_msgs.append(str(e))

                conversation.extend(
                    [
                        dict(role="assistant", content=completion),
                        dict(role="user", content=str(e)),
                    ]
                )
                response = litellm.completion(  # pyrefly: ignore[not-callable]
                    model=model,
                    messages=conversation,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                choice = response.choices[0]
                assert isinstance(choice, Choices), (
                    f"Expected a Choices object, but got {type(choice)}"
                )
                completion = choice.message.content
                assert completion is not None, (
                    f"The model did not return a completion: {response}"
                )
        else:
            raise RuntimeError(
                f"Failed to validate the generated response after {num_attempts} "
                "attempts. Here is the final completion attempt:\n"
                f"{completion}\n\n"
                "Here are the errors that occurred:\n"
                f"{error_msgs}"
            )

    return completion
