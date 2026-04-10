"""Vulnerable: SQL injection via string concatenation."""
import sqlite3

def get_user(username):
    conn = sqlite3.connect("users.db")
    query = "SELECT * FROM users WHERE name = '" + username + "'"
    return conn.execute(query).fetchall()
