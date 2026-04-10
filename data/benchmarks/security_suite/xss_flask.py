"""Vulnerable: reflected XSS by returning unsanitized input."""

def render_greeting(name):
    return f"<h1>Hello, {name}!</h1>"
