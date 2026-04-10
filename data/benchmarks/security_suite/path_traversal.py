"""Vulnerable: path traversal via unsanitized user input."""
import os

def read_file(filename):
    base = "/var/www/uploads"
    path = os.path.join(base, filename)
    with open(path) as f:
        return f.read()
