#!/bin/bash
set -e

WEBSHOP_DIR=/app/workspace/agent_system/environments/env_package/webshop/webshop

# Symlink webshop data if not already present on the host
if [ ! -f "$WEBSHOP_DIR/data/items_shuffle_1000.json" ]; then
    rm -rf "$WEBSHOP_DIR/data" 2>/dev/null || true
    ln -sfn /opt/webshop-data "$WEBSHOP_DIR/data"
fi

# Symlink search engine indexes if not already present on the host
mkdir -p "$WEBSHOP_DIR/search_engine"
for idx in indexes indexes_1k indexes_100 indexes_100k; do
    if [ ! -d "$WEBSHOP_DIR/search_engine/$idx" ]; then
        ln -sfn "/opt/webshop-$idx" "$WEBSHOP_DIR/search_engine/$idx"
    fi
done

# Re-install verl-agent from mounted workspace (fast, no-deps)
if [ -f /app/workspace/setup.py ]; then
    pip install --no-deps -e /app/workspace 2>&1 | tail -1
fi

exec "$@"
