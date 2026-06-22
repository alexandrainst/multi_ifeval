"""Tests for the `data_models` module."""

from multi_ifeval.data_models import GeneratedExample, Kwarg


def test_generated_example_allows_empty_list_kwarg_value() -> None:
    """A kwarg whose value is an empty list must not crash validation.

    Regression test: ``model_post_init`` filtered empty kwargs with a membership
    test against a set literal ``{"", []}``. Building that set raised
    ``TypeError: unhashable type: 'list'`` whenever the test was evaluated, so any
    example carrying a non-empty kwarg list failed. The fix uses a tuple.
    """
    example = GeneratedExample(
        new_prompt="some prompt",
        new_instruction_id_list=["keywords:existence", "punctuation:no_comma"],
        new_kwargs=[
            [Kwarg(name="keywords", value=["alpha", "beta"]), Kwarg(name="extra", value=[])],
            [Kwarg(name="blank", value="")],
        ],
    )
    # empty-string and empty-list kwargs are filtered out; real values are kept
    assert example.new_kwargs[0] == [Kwarg(name="keywords", value=["alpha", "beta"])]
    assert example.new_kwargs[1] == []
