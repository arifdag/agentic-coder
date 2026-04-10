"""Phantom import that looks similar to a real package."""
import pandaz

def load_csv(path):
    return pandaz.read_csv(path)
