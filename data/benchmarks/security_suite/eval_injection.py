"""Vulnerable: code injection via eval on user input."""

def compute(expression):
    return eval(expression)
