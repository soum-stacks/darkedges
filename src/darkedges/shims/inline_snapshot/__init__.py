"""DARKEDGES SHIM for the `inline-snapshot` package (no wheel in the competition wheelhouse).
Comparison semantics only: snapshot(x) behaves as x. No updating/recording. Used ONLY when the real package is absent."""
__shim__ = True


class Snapshot:          # used for type annotations in fastapi tests/utils.py
    pass


def snapshot(value=None):
    return value


def Is(value):
    return value
