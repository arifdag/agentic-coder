"""Mixes real and phantom packages."""
import requests
import fasthelpers

def fetch(url):
    resp = requests.get(url)
    return fasthelpers.sanitize(resp.text)
