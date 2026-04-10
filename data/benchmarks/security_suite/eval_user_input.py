"""Deliberately unsafe: eval on user input (for SAST evaluation only)."""


def compute(expr: str):
    return eval(expr)
