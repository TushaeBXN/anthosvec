"""Safe filter expressions over document fields.

Filters are Python-syntax boolean expressions evaluated against a doc's
fields, e.g. ``"price < 100 and category == 'books'"``. Only comparisons,
boolean operators, membership tests, field names, and literals are allowed —
no calls, attributes, subscripts, or dunder access.
"""

from __future__ import annotations

import ast
from typing import Any, Callable, Dict

_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.UnaryOp, ast.Compare, ast.Name, ast.Load,
    ast.Constant, ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt,
    ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.List, ast.Tuple,
    ast.USub, ast.UAdd,
)


def compile_filter(expression: str) -> Callable[[Dict[str, Any]], bool]:
    tree = ast.parse(expression, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"unsupported syntax in filter: {type(node).__name__} "
                f"(filters allow comparisons, and/or/not, in, and literals)")
    code = compile(tree, "<filter>", "eval")

    def evaluate(fields: Dict[str, Any]) -> bool:
        try:
            return bool(eval(code, {"__builtins__": {}}, dict(fields)))
        except NameError:
            # a referenced field is absent on this doc -> doesn't match
            return False
        except TypeError:
            # e.g. comparing None with int -> doesn't match
            return False

    return evaluate
