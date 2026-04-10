"""Imports a phantom package alongside valid ones."""
import os
import json
import superturboparser

def process(data):
    return superturboparser.parse(json.dumps(data))
