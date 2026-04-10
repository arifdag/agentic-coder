"""Vulnerable: OS command injection via shell=True."""
import subprocess

def ping_host(host):
    return subprocess.check_output("ping -c 1 " + host, shell=True)
