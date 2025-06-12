from typing import Callable

def register_classes(registry: dict[str, type]) -> Callable[[type], type]:
    """
    Return a decorator that will add any decorated class into `registry`
    under its class name.
    """
    def decorator(cls: type) -> type:
        registry[cls.__name__] = cls
        return cls
    return decorator