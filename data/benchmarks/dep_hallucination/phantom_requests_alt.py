import json
import reqwests  # noqa: F401 — phantom package for dependency gate evaluation


def echo(x):
    return json.dumps(x)
