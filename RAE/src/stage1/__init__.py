__all__ = ["RAE"]


def __getattr__(name):
    if name == "RAE":
        from .rae import RAE

        return RAE
    raise AttributeError(name)
