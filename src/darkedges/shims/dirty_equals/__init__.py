"""DARKEDGES SHIM for the `dirty-equals` package (no wheel in the competition wheelhouse).
Minimal re-implementation of the matchers used by test suites; semantics are simplified. Used ONLY when the real package is absent."""
import re as _re
import uuid as _uuid

__shim__ = True


def _eq(a, b):
    return a == b


class DirtyEquals:
    def __init__(self, *args, **kwargs):
        self._args, self._kwargs = args, kwargs

    def equals(self, other):
        raise NotImplementedError

    def __eq__(self, other):
        return bool(self.equals(other))

    def __ne__(self, other):
        return not self.__eq__(other)

    __hash__ = None

    def __or__(self, other):
        return _Or(self, other)

    def __and__(self, other):
        return _And(self, other)

    def __invert__(self):
        return _Not(self)

    def __repr__(self):
        return f"{type(self).__name__}({', '.join(map(repr, self._args))})"


class _Or(DirtyEquals):
    def equals(self, other):
        return any(_eq(other, a) for a in self._args)

    def __init__(self, a, b):
        super().__init__(a, b)


class _And(_Or):
    def equals(self, other):
        return all(_eq(other, a) for a in self._args)


class _Not(DirtyEquals):
    def equals(self, other):
        return not _eq(other, self._args[0])


class AnyThing(DirtyEquals):
    def equals(self, other):
        return True


class IsOneOf(DirtyEquals):
    def equals(self, other):
        return any(_eq(other, a) for a in self._args)


class IsInstance(DirtyEquals):
    def equals(self, other):
        return isinstance(other, self._args[0])


class IsDict(DirtyEquals):
    """IsDict(expected_dict) / IsDict(k=v); no args -> any dict. Values may themselves be matchers."""
    partial = False

    def __init__(self, expected=None, **kwargs):
        exp = dict(expected or {})
        exp.update(kwargs)
        super().__init__(exp)
        self.expected = exp

    def equals(self, other):
        if not isinstance(other, dict):
            return False
        if not self.expected:
            return True
        if self.partial:
            return all(k in other and _eq(other[k], v) for k, v in self.expected.items())
        return set(other) == set(self.expected) and all(_eq(other[k], v) for k, v in self.expected.items())


class IsPartialDict(IsDict):
    partial = True


IsIgnoreDict = IsPartialDict
IsStrictDict = IsDict


class IsList(DirtyEquals):
    def __init__(self, *items, check_order=True, length=None):
        super().__init__(*items)
        self.check_order, self.length = check_order, length

    def equals(self, other):
        if not isinstance(other, (list, tuple)):
            return False
        if self.length is not None and len(other) != self.length:
            return False
        if not self._args:
            return True
        if len(other) != len(self._args):
            return False
        if self.check_order:
            return all(_eq(o, a) for o, a in zip(other, self._args))
        rest = list(other)
        for a in self._args:
            for i, o in enumerate(rest):
                if _eq(o, a):
                    del rest[i]
                    break
            else:
                return False
        return True


IsTuple = IsList


class HasLen(DirtyEquals):
    def equals(self, other):
        try:
            return len(other) == self._args[0]
        except TypeError:
            return False


class Contains(DirtyEquals):
    def equals(self, other):
        try:
            return all(a in other for a in self._args)
        except TypeError:
            return False


class IsStr(DirtyEquals):
    def __init__(self, regex=None, *, min_length=None, max_length=None, case=None):
        super().__init__(regex)
        self.regex, self.min_length, self.max_length = regex, min_length, max_length

    def equals(self, other):
        if not isinstance(other, str):
            return False
        if self.regex is not None and not _re.fullmatch(self.regex, other):
            return False
        return (self.min_length is None or len(other) >= self.min_length) and \
               (self.max_length is None or len(other) <= self.max_length)


class IsBytes(DirtyEquals):
    def equals(self, other):
        return isinstance(other, bytes)


class IsAnyStr(DirtyEquals):
    def equals(self, other):
        return isinstance(other, (str, bytes))


class IsInt(DirtyEquals):
    def __init__(self, *, gt=None, ge=None, lt=None, le=None):
        super().__init__()
        self.b = (gt, ge, lt, le)

    def equals(self, other):
        if isinstance(other, bool) or not isinstance(other, int):
            return False
        gt, ge, lt, le = self.b
        return (gt is None or other > gt) and (ge is None or other >= ge) and \
               (lt is None or other < lt) and (le is None or other <= le)


class IsPositiveInt(IsInt):
    def __init__(self):
        super().__init__(gt=0)


class IsFloat(DirtyEquals):
    def equals(self, other):
        return isinstance(other, float)


class IsNumeric(DirtyEquals):
    def equals(self, other):
        return isinstance(other, (int, float)) and not isinstance(other, bool)


IsNumber = IsNumeric


class IsBool(DirtyEquals):
    def equals(self, other):
        return isinstance(other, bool)


class IsTrueLike(DirtyEquals):
    def equals(self, other):
        return bool(other)


class IsFalseLike(DirtyEquals):
    def equals(self, other):
        return not bool(other)


class IsUUID(DirtyEquals):
    def equals(self, other):
        try:
            _uuid.UUID(str(other))
            return True
        except ValueError:
            return False


class IsUrl(DirtyEquals):
    def equals(self, other):
        return isinstance(other, str) and bool(_re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", other))


class IsNow(DirtyEquals):          # simplified: any datetime / ISO-like string
    def equals(self, other):
        import datetime as _dt
        return isinstance(other, (_dt.datetime, str, int, float))
