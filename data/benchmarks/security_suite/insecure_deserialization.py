"""Vulnerable: pickle deserialization of untrusted data."""
import pickle

def load_data(raw_bytes):
    return pickle.loads(raw_bytes)
