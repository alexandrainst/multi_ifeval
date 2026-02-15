"""Data models used in the project."""

import typing as t

from pydantic import BaseModel


class Example(BaseModel):
    """A dataset sample from an instruction-following dataset."""

    key: int
    prompt: str
    instruction_id_list: list[str]
    kwargs: list[dict[str, t.Any]]
