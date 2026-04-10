"""Vulnerable: unvalidated redirect URL."""

def redirect_url(target):
    return f"HTTP/1.1 302 Found\r\nLocation: {target}\r\n\r\n"
