#!/usr/bin/env python3
"""Self-healing integrity guard for the claim graph.

The design's central write-side invariant (docs/graphbuilding-fusion-design.md §7):
the engine must never be able to commit drift. The debt doc found four drift
species in a live session (103 dangling, 134 chains, 13 self-loops, 8 orphans) —
all because the write path never held the invariants the read path assumed.

`validate(conn)` reports those four species. `heal(conn)` fixes them through the
one shared `root()` helper (path-compress, break cycles, resolve dangling) and
reconciles `is_canonical` — it never throws on bad data.

CLI:
    uv run python scripts/graph-validate.py            # report; exit 1 if drift
    uv run python scripts/graph-validate.py --heal     # fix in place, then report
    uv run python scripts/graph-validate.py --json

Wired into the pre-commit `verifier` agent so drift can never be committed.

An entity is canonical iff `canonical_id IS NULL` (the real derived-db schema has
no `is_canonical` column).

Drift species:
  dangling   entity.canonical_id points to a row that does not exist
  chain      entity.canonical_id points to a row that is itself a variant (multi-hop)
  self_loop  entity.canonical_id == id
  orphan     a claim whose subject/object entity id does not exist
             (report-only — heal cannot fabricate the missing entity)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph_db  # noqa: E402  (shared connect/root — never inline COALESCE)

DEFAULT_DB = Path(__file__).resolve().parent.parent / ".vault-meta" / "graph" / "graph.db"

SPECIES = ("dangling", "chain", "self_loop", "orphan")


def _has_table(conn, name):
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def validate(conn):
    """Return a drift report: per-species counts, a total, and offending ids."""
    rows = conn.execute("SELECT id, canonical_id FROM entities").fetchall()
    live = {r[0] for r in rows}
    parent_of = {i: c for i, c in rows}

    found = {s: [] for s in SPECIES}
    for i, canonical_id in rows:
        if canonical_id is None:
            continue  # canonical root — fine
        if canonical_id == i:
            found["self_loop"].append(i)
        elif canonical_id not in live:
            found["dangling"].append(i)
        elif parent_of.get(canonical_id) is not None:
            # parent is itself a variant => multi-hop chain
            found["chain"].append(i)

    # orphan: claims referencing a non-existent entity (report-only)
    if _has_table(conn, "claims"):
        for (cid,) in conn.execute(
            "SELECT id FROM claims "
            "WHERE subject_entity_id NOT IN (SELECT id FROM entities) "
            "   OR object_entity_id  NOT IN (SELECT id FROM entities)"
        ).fetchall():
            found["orphan"].append(cid)

    report = {s: len(found[s]) for s in SPECIES}
    report["total"] = sum(report.values())
    report["_ids"] = found
    return report


def heal(conn):
    """Fix the pointer-drift class in place (dangling / chain / self-loop / cycle).

    Runs the shared root() on every variant: path-compresses chains, breaks
    cycles and self-loops, and lands dangling pointers on the last live id.
    Idempotent; never throws. Orphan claims are NOT touched — heal cannot
    fabricate a missing entity, so those stay reported for human/ingest repair.
    """
    variants = [
        r[0]
        for r in conn.execute(
            "SELECT id FROM entities WHERE canonical_id IS NOT NULL"
        ).fetchall()
    ]
    for eid in variants:
        graph_db.root(conn, eid, compress=True)
    conn.commit()


def _format(report):
    if report["total"] == 0:
        return "✓ graph integrity clean — 0 dangling / 0 chain / 0 self-loop / 0 orphan"
    lines = [f"⚠ {report['total']} integrity issue(s):"]
    for s in SPECIES:
        if report[s]:
            sample = report["_ids"][s][:8]
            lines.append(f"  {s:10} {report[s]:5}  ids={sample}{' …' if report[s] > 8 else ''}")
    lines.append("  run with --heal to fix in place")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Claim-graph integrity guard")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="derived graph db path")
    parser.add_argument("--heal", action="store_true", help="fix drift in place")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database not found at {db_path}", file=sys.stderr)
        print("Run scripts/graph-build.py first.", file=sys.stderr)
        sys.exit(1)

    conn = graph_db.connect(db_path)
    report = validate(conn)
    healed = False
    if args.heal and report["total"] > 0:
        heal(conn)
        report = validate(conn)
        healed = True

    if args.json:
        out = {s: report[s] for s in SPECIES}
        out["total"] = report["total"]
        out["healed"] = healed
        print(json.dumps(out))
    else:
        if healed:
            print("healed drift; re-validated:")
        print(_format(report))

    conn.close()
    sys.exit(0 if report["total"] == 0 else 1)


if __name__ == "__main__":
    main()
