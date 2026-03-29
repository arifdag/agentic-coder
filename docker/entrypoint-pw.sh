#!/usr/bin/env bash
# Entrypoint for the Playwright sandbox container.
# If an index.html exists in /workspace/site/, start a local HTTP server
# so that tests can hit http://localhost:8080.

set -e

SITE_DIR="/workspace/site"

if [ -d "$SITE_DIR" ] && [ -f "$SITE_DIR/index.html" ]; then
    python3 -m http.server 8080 --directory "$SITE_DIR" &
    SERVER_PID=$!

    for i in $(seq 1 20); do
        if python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080')" 2>/dev/null; then
            break
        fi
        sleep 0.25
    done

    export BASE_URL="${BASE_URL:-http://localhost:8080}"
fi

exec "$@"
