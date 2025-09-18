from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar


T = TypeVar("T")
R = TypeVar("R")


def run_tasks(
    items: Iterable[T],
    func: Callable[[T], R],
    *,
    max_workers: int,
) -> list[R]:
    """Apply ``func`` to each item, with optional thread pooling."""

    items = list(items)
    if max_workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(func, item) for item in items]
        return [future.result() for future in futures]
