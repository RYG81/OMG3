"""Small AST-whitelist evaluator for numeric ComfyUI-OMG expressions."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable

_BINARY: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def safe_eval_math(
    expression: str,
    variables: dict[str, float],
    functions: dict[str, Callable[..., float]],
    constants: dict[str, float] | None = None,
) -> float:
    """Evaluate arithmetic using only approved variables/functions/constants."""

    if len(expression) > 512:
        raise ValueError("Expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError("Invalid expression syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > 100:
        raise ValueError("Expression is too complex")

    names = dict(constants or {})
    names.update(variables)

    def evaluate(node: ast.AST, depth: int = 0) -> float:
        if depth > 20:
            raise ValueError("Expression nesting is too deep")
        if isinstance(node, ast.Expression):
            return evaluate(node.body, depth + 1)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in names:
            return float(names[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            left = evaluate(node.left, depth + 1)
            right = evaluate(node.right, depth + 1)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent is too large")
            return float(_BINARY[type(node.op)](left, right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return float(_UNARY[type(node.op)](evaluate(node.operand, depth + 1)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = functions.get(node.func.id)
            if function is None or node.keywords:
                raise ValueError(f"Function {node.func.id!r} is not allowed")
            args = [evaluate(argument, depth + 1) for argument in node.args]
            return float(function(*args))
        raise ValueError(f"Expression element {type(node).__name__} is not allowed")

    result = evaluate(tree)
    if not math.isfinite(result):
        raise ValueError("Expression result must be finite")
    return result
