#!/usr/bin/env bash
# roundtrip_live.sh — Stage B helper for AC5
#
# Verifies the round-trip stays byte-equal on the CURRENT wiki/graph/ vault:
#   1. Copy the live derived db to a temp location (never touch the live db).
#   2. Run graph-export.py to write a fresh export into a temp vault dir.
#   3. Run graph-build.py to rebuild a derived db from that temp vault.
#   4. Compare all 9 tables row-for-row between the original live db copy
#      and the freshly rebuilt db.
#
# Usage:
#   bash tests/roundtrip_live.sh
#
# Exit 0 on pass (all 9 tables byte-equal), non-zero on any failure.
#
# Prerequisites: .vault-meta/graph/graph.db must exist (run graph-build.py first).
# Does NOT depend on ~/.graphbuilding/graph.db.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIVE_DB="$REPO_ROOT/.vault-meta/graph/graph.db"
SCRIPTS="$REPO_ROOT/scripts"

if [ ! -f "$LIVE_DB" ]; then
    echo "ERROR: derived db not found at $LIVE_DB" >&2
    echo "Run: uv run python scripts/graph-build.py wiki/graph .vault-meta/graph/graph.db" >&2
    exit 1
fi

# Create a temp workspace
TMPDIR_WORK="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_WORK"' EXIT

LIVE_DB_COPY="$TMPDIR_WORK/live_copy.db"
TEMP_VAULT="$TMPDIR_WORK/vault"
REBUILT_DB="$TMPDIR_WORK/rebuilt.db"

mkdir -p "$TEMP_VAULT"

echo "==> Copying live db to temp location..."
cp "$LIVE_DB" "$LIVE_DB_COPY"

echo "==> Running graph-export.py on the live db copy..."
uv run python "$SCRIPTS/graph-export.py" "$LIVE_DB_COPY" "$TEMP_VAULT" >/dev/null

echo "==> Running graph-build.py from temp vault..."
uv run python "$SCRIPTS/graph-build.py" "$TEMP_VAULT" "$REBUILT_DB" >/dev/null

echo "==> Comparing tables row-for-row..."
uv run python - "$LIVE_DB_COPY" "$REBUILT_DB" <<'PYEOF'
import sqlite3
import sys

src_path, dst_path = sys.argv[1], sys.argv[2]
src = sqlite3.connect(src_path)
dst = sqlite3.connect(dst_path)

TABLES = [
    "papers", "sections", "paper_authors", "entities",
    "predicates", "claims", "entity_edges", "citation_links", "aliases",
]

all_pass = True
for table in TABLES:
    try:
        src_rows = set(src.execute(f"SELECT * FROM {table}").fetchall())
        dst_rows = set(dst.execute(f"SELECT * FROM {table}").fetchall())
    except Exception as e:
        print(f"  FAIL {table}: query error: {e}")
        all_pass = False
        continue

    if src_rows == dst_rows:
        print(f"  PASS {table}: {len(src_rows)} rows byte-equal")
    else:
        only_src = src_rows - dst_rows
        only_dst = dst_rows - src_rows
        print(f"  FAIL {table}: src={len(src_rows)} dst={len(dst_rows)}")
        if only_src:
            sample = list(only_src)[:3]
            print(f"         only-in-src (first 3): {sample}")
        if only_dst:
            sample = list(only_dst)[:3]
            print(f"         only-in-dst (first 3): {sample}")
        all_pass = False

if all_pass:
    print("\n  Round-trip PASS: all 9 tables byte-equal.")
    sys.exit(0)
else:
    print("\n  Round-trip FAIL: see diffs above.")
    sys.exit(1)
PYEOF
