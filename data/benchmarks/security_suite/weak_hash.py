"""Vulnerable: MD5 used for password hashing."""
import hashlib

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()
