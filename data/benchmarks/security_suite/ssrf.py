"""Vulnerable: server-side request forgery via user-supplied URL."""
import urllib.request

def fetch(url):
    return urllib.request.urlopen(url).read()
