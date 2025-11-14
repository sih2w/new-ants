from typing import TypedDict, TypeVar


T = TypeVar("T")


class Vector2(TypedDict):
    X: T
    Y: T