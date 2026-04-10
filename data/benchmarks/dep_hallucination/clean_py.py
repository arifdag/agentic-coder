"""All valid imports -- should pass cleanly."""
import os
import json
import sys

def info():
    return {"pid": os.getpid(), "version": sys.version}
