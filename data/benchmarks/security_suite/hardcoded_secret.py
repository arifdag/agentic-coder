"""Vulnerable: hardcoded credential in source."""
import hashlib

API_KEY = "sk-live-ABC123secretKEY456"

def authenticate(token):
    return hashlib.sha256(token.encode()).hexdigest() == hashlib.sha256(API_KEY.encode()).hexdigest()
