"""Apply safe entity-resolution merges — write loser frontmatter from a vetted TSV list.

DRY-RUN is the default: prints the computed plan (loser->winner, degrees, reason)
and exits 0 without writing anything.  Writes only with --commit.

Usage:
    # Dry-run (default) — print plan, write nothing
    uv run python scripts/graph-resolve-apply.py \
        --merge-file specs/merges/graph-resolve-apply.tsv

    # Live commit — write 9 loser .md frontmatter fields
    uv run python scripts/graph-resolve-apply.py \
        --merge-file specs/merges/graph-resolve-apply.tsv --commit

    # JSON output (dry-run)
    uv run python scripts/graph-resolve-apply.py \
        --merge-file specs/merges/graph-resolve-apply.tsv --json

Functional requirements (spec §7):
  FR2  winner = argmax(claim-degree on pre-merge db); tie -> min id
  FR3  pairs deduped by frozenset({a,b}); multi-id groups flatten to one winner
  FR4  merge_confidence set on loser from TSV reason (1.0 for T1 exact; Jaccard for T2)
  FR5  default dry-run; writes only with --commit (BR6)
  FR6  skip if id missing, loser already merged elsewhere, or winner==loser; non-zero
       only if ZERO pairs apply
  FR7  operator re-runs graph-export + graph-build separately (not done here)
  FR10 merge list lives in --merge-file, not hardcoded

Stage A: dry-run + sandbox tests.  Stage B: live --commit (needs human App-A ratification).
"""

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from graph_db import root  # noqa: E402

# Default paths
DEFAULT_DB = Path(__file__).resolve().parent.parent / ".vault-meta" / "graph" / "graph.db"
DEFAULT_ENTITIES_DIR = Path(__file__).resolve().parent.parent / "wiki" / "graph" / "entities"


# ---------------------------------------------------------------------------
# TSV loader
# ---------------------------------------------------------------------------

def _load_merge_file(path: Path) -> list:
    """Load a TSV merge file, skipping comment lines.

    Returns list of (loser_id: int, winner_id: int, reason: str).
    Exits 1 if file is missing or has zero data rows.
    """
    if not path.exists():
        print(f"ERROR: merge file not found: {path}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split("\t", 2)
        if len(parts) < 2:
            continue
        try:
            loser_id = int(parts[0])
            winner_id = int(parts[1])
            reason = parts[2].strip() if len(parts) > 2 else ""
            rows.append((loser_id, winner_id, reason))
        except ValueError:
            continue

    if not rows:
        print("ERROR: merge file has no approved pairs to apply.", file=sys.stderr)
        sys.exit(1)

    return rows


# ---------------------------------------------------------------------------
# Degree lookup
# ---------------------------------------------------------------------------

def _claim_degree(conn, entity_id: int) -> int:
    """Count claims where entity appears as subject OR object (pre-merge, root-resolved)."""
    row = conn.execute(
        "SELECT COUNT(*) FROM claims "
        "WHERE subject_entity_id = ? OR object_entity_id = ?",
        (entity_id, entity_id),
    ).fetchone()
    return row[0] if row else 0


def _entity_exists(conn, entity_id: int) -> bool:
    row = conn.execute("SELECT 1 FROM entities WHERE id = ?", (entity_id,)).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Winner selection (spec §4)
# ---------------------------------------------------------------------------

def _select_winner(conn, id_a: int, id_b: int) -> tuple:
    """Return (winner, loser) by claim-degree; tie -> min id."""
    deg_a = _claim_degree(conn, id_a)
    deg_b = _claim_degree(conn, id_b)
    if deg_a > deg_b:
        return id_a, id_b, deg_a, deg_b
    elif deg_b > deg_a:
        return id_b, id_a, deg_b, deg_a
    else:
        # Tie: min id wins
        winner = min(id_a, id_b)
        loser = max(id_a, id_b)
        return winner, loser, deg_a, deg_b


# ---------------------------------------------------------------------------
# Confidence from reason string
# ---------------------------------------------------------------------------

def _confidence_from_reason(reason: str) -> float:
    """Infer merge_confidence from reason string.

    T1 exact -> 1.0.  T2 with J=X.XXX -> float(X.XXX).  Otherwise 1.0.
    """
    # Jaccard score in reason
    m = re.search(r"J=([0-9]+\.[0-9]+)", reason)
    if m:
        return float(m.group(1))
    # T1 exact match
    if "T1" in reason or "exact" in reason.lower():
        return 1.0
    return 1.0


# ---------------------------------------------------------------------------
# Plan builder — dedup, flatten groups, winner resolution
# ---------------------------------------------------------------------------

def _build_plan(conn, raw_rows: list) -> list:
    """Build the merge plan from raw TSV rows.

    Returns list of dicts:
      {loser, winner, deg_loser, deg_winner, reason, skip_reason}

    Skipped entries have skip_reason set; they are not written.
    Deduplicated by frozenset({a,b}).  Multi-id groups (e.g. CFG 730/1354/1386)
    flatten to one group winner.
    """
    # Step 1: dedup raw pairs by frozenset
    seen_pairs = set()
    deduped = []
    for loser_raw, winner_raw, reason in raw_rows:
        key = frozenset({loser_raw, winner_raw})
        if key in seen_pairs:
            continue
        if loser_raw == winner_raw:
            continue  # self-pair, skip immediately
        seen_pairs.add(key)
        deduped.append((loser_raw, winner_raw, reason))

    # Step 2: for each pair compute winner by degree
    pair_info = []  # (winner, loser, deg_winner, deg_loser, reason)
    for a, b, reason in deduped:
        winner, loser, deg_w, deg_l = _select_winner(conn, a, b)
        pair_info.append((winner, loser, deg_w, deg_l, reason))

    # Step 3: build group membership — all ids that share any pair
    # Use union-find to find group winners
    parent = {}

    def _find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = _find(parent[x])
        return parent[x]

    def _union(x, y):
        rx, ry = _find(x), _find(y)
        if rx != ry:
            parent[ry] = rx  # x's root wins

    # First pass: union all pairs and identify group-level winner
    group_winner = {}  # representative -> group-level winner (highest degree / min-id)
    # Collect all pairs with their degrees to determine group winner
    all_ids_degrees = {}
    for winner, loser, deg_w, deg_l, reason in pair_info:
        all_ids_degrees[winner] = deg_w
        if loser not in all_ids_degrees:
            all_ids_degrees[loser] = deg_l
        _union(winner, loser)

    # Determine the group winner for each component
    # Group winner = max degree in group; tie -> min id
    group_members = {}
    for eid in all_ids_degrees:
        rep = _find(eid)
        if rep not in group_members:
            group_members[rep] = []
        group_members[rep].append(eid)

    final_group_winner = {}
    for rep, members in group_members.items():
        best = None
        best_deg = -1
        for m in members:
            d = all_ids_degrees.get(m, 0)
            if d > best_deg or (d == best_deg and best is not None and m < best):
                best_deg = d
                best = m
        for m in members:
            final_group_winner[m] = best

    # Step 4: emit plan with flattened winners
    plan = []
    for winner_raw, loser_raw, deg_w_raw, deg_l_raw, reason in pair_info:
        group_w = final_group_winner.get(winner_raw, winner_raw)
        group_w = final_group_winner.get(loser_raw, group_w)
        actual_winner = group_w

        # After flattening, recompute loser/winner: the non-group-winner is loser
        a, b = winner_raw, loser_raw
        if a == actual_winner:
            actual_loser = b
        elif b == actual_winner:
            actual_loser = a
        else:
            # Both are non-winners — this shouldn't happen in our data
            actual_loser = loser_raw

        if actual_winner == actual_loser:
            continue  # self-pair after flattening

        # Recompute degrees for the final (winner, loser) assignment
        deg_winner = all_ids_degrees.get(actual_winner, 0)
        deg_loser = all_ids_degrees.get(actual_loser, 0)

        plan.append({
            "loser": actual_loser,
            "winner": actual_winner,
            "deg_loser": deg_loser,
            "deg_winner": deg_winner,
            "reason": reason,
        })

    # Dedup plan by (loser, winner) — a loser can only go to one winner
    seen_losers = {}
    final_plan = []
    for entry in plan:
        loser = entry["loser"]
        if loser in seen_losers:
            continue  # already assigned
        seen_losers[loser] = entry["winner"]
        final_plan.append(entry)

    return final_plan


# ---------------------------------------------------------------------------
# Entity .md frontmatter editor
# ---------------------------------------------------------------------------

def _find_entity_md(entities_dir: Path, entity_id: int) -> Path | None:
    """Find the entity .md file by id suffix (e.g. *__e1366.md)."""
    pattern = f"*__e{entity_id}.md"
    matches = list(entities_dir.glob(pattern))
    if not matches:
        return None
    return matches[0]


def _winner_slug(entities_dir: Path, winner_id: int) -> str | None:
    """Return the slug (filename stem) of the winner entity."""
    md = _find_entity_md(entities_dir, winner_id)
    if md is None:
        return None
    return md.stem  # e.g. "clip-vision-encoder__e1368"


def _read_frontmatter(text: str) -> tuple:
    """Parse YAML frontmatter from a markdown string.

    Returns (frontmatter_text, body_text) where frontmatter_text is the raw
    YAML between the first two '---' delimiters (without delimiters).
    If no frontmatter, returns ("", text).
    """
    if not text.startswith("---"):
        return "", text
    # Find closing ---
    rest = text[3:]
    close_idx = rest.find("\n---")
    if close_idx == -1:
        return "", text
    fm = rest[:close_idx]
    body = rest[close_idx + 4:]  # skip \n---
    return fm, body


def _update_frontmatter_field(fm_text: str, field: str, value) -> str:
    """Update or add a field in raw YAML frontmatter text.

    Preserves all other lines verbatim. Handles null/None specially.
    """
    lines = fm_text.split("\n")
    yaml_value = _to_yaml_scalar(value)
    new_line = f"{field}: {yaml_value}"
    updated = False
    result = []
    for line in lines:
        if re.match(rf"^{re.escape(field)}\s*:", line):
            result.append(new_line)
            updated = True
        else:
            result.append(line)
    if not updated:
        result.append(new_line)
    return "\n".join(result)


def _to_yaml_scalar(value) -> str:
    """Convert a Python value to a YAML scalar string (inline, no block style)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return str(value)
    if isinstance(value, int):
        return str(value)
    # String — quote if needed
    if isinstance(value, str):
        # Check if quoting is needed (contains special chars or looks ambiguous)
        if any(c in value for c in (":", "#", "{", "}", "[", "]", ",", "&", "*", "?", "|", "<", ">", "=", "!", "%", "@", "`")):
            return f"'{value}'"
        return value
    return str(value)


def _write_loser_md(md_path: Path, winner_id: int, winner_slug: str, confidence: float) -> str:
    """Edit loser entity .md frontmatter in-place.

    Sets:
      is_canonical: false
      canonical_id: <winner_id>
      canonical: '[[<winner_slug>]]'
      merge_confidence: <confidence>

    Returns "written" | "already_applied" | "already_merged_elsewhere"
    """
    text = md_path.read_text()
    fm_text, body = _read_frontmatter(text)
    if not fm_text and not body:
        return "no_frontmatter"

    # Check existing canonical_id
    existing_cid = None
    for line in fm_text.split("\n"):
        m = re.match(r"^canonical_id\s*:\s*(.+)$", line.strip())
        if m:
            val = m.group(1).strip()
            if val not in ("null", "~", ""):
                try:
                    existing_cid = int(val)
                except ValueError:
                    pass

    if existing_cid is not None:
        if existing_cid == winner_id:
            return "already_applied"
        else:
            return "already_merged_elsewhere"

    # Apply edits
    fm_text = _update_frontmatter_field(fm_text, "is_canonical", False)
    fm_text = _update_frontmatter_field(fm_text, "canonical_id", winner_id)
    fm_text = _update_frontmatter_field(fm_text, "canonical", f"[[{winner_slug}]]")
    fm_text = _update_frontmatter_field(fm_text, "merge_confidence", confidence)

    new_text = f"---\n{fm_text}\n---{body}"
    md_path.write_text(new_text)
    return "written"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Apply safe entity-resolution merges (dry-run by default)."
    )
    p.add_argument(
        "--merge-file", required=True, type=Path,
        help="TSV file with loser_id<TAB>winner_id<TAB>reason rows.",
    )
    p.add_argument(
        "--db", type=Path, default=DEFAULT_DB,
        help="Path to the derived graph.db (default: .vault-meta/graph/graph.db).",
    )
    p.add_argument(
        "--entities-dir", type=Path, default=DEFAULT_ENTITIES_DIR,
        help="Path to wiki/graph/entities/ directory (default: auto from repo root).",
    )
    p.add_argument(
        "--commit", action="store_true",
        help="Actually write loser .md frontmatter. Default is dry-run (no writes).",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Emit JSON output: {dry_run: bool, plan: [...]}.",
    )
    args = p.parse_args(argv)

    # Validate db
    if not args.db.exists():
        msg = (
            f"Derived db not found: {args.db}\n"
            "Run graph-build.py first to create the derived db."
        )
        print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)

    import sqlite3 as _sqlite3
    conn = _sqlite3.connect(str(args.db))

    # Load merge file
    raw_rows = _load_merge_file(args.merge_file)

    # Validate ids exist in db; skip ghosts
    validated = []
    skipped_missing = []
    for loser_raw, winner_raw, reason in raw_rows:
        a_exists = _entity_exists(conn, loser_raw)
        b_exists = _entity_exists(conn, winner_raw)
        if not a_exists or not b_exists:
            missing = []
            if not a_exists:
                missing.append(str(loser_raw))
            if not b_exists:
                missing.append(str(winner_raw))
            skipped_missing.append((loser_raw, winner_raw, missing))
            print(
                f"SKIP pair ({loser_raw}, {winner_raw}): id(s) {missing} not in db.",
                file=sys.stderr,
            )
            continue
        validated.append((loser_raw, winner_raw, reason))

    if not validated:
        print("ERROR: no approved pairs to apply (all skipped or missing).", file=sys.stderr)
        sys.exit(1)

    # Build plan
    plan = _build_plan(conn, validated)
    conn.close()

    dry_run = not args.commit

    if args.json:
        print(json.dumps({"dry_run": dry_run, "plan": plan}, indent=2))
        return

    # Human-readable output
    mode_str = "DRY-RUN" if dry_run else "COMMIT"
    print(f"\n  [{mode_str}] graph-resolve-apply — {len(plan)} merge(s) planned\n")
    for entry in plan:
        print(
            f"  e{entry['loser']} → e{entry['winner']}"
            f"  (deg {entry['deg_loser']} < {entry['deg_winner']})"
            f"  — {entry['reason']}"
        )

    if dry_run:
        print(f"\n  DRY-RUN complete. Re-run with --commit to persist.")
        return

    # --- COMMIT MODE ---
    entities_dir = args.entities_dir
    if not entities_dir.exists():
        print(f"ERROR: entities dir not found: {entities_dir}", file=sys.stderr)
        sys.exit(1)

    written_count = 0
    for entry in plan:
        loser_id = entry["loser"]
        winner_id = entry["winner"]
        confidence = _confidence_from_reason(entry["reason"])

        md_path = _find_entity_md(entities_dir, loser_id)
        if md_path is None:
            print(
                f"  SKIP e{loser_id}: .md file not found in {entities_dir}",
                file=sys.stderr,
            )
            continue

        w_slug = _winner_slug(entities_dir, winner_id)
        if w_slug is None:
            print(
                f"  SKIP e{loser_id}: winner e{winner_id} .md not found for wikilink.",
                file=sys.stderr,
            )
            continue

        result = _write_loser_md(md_path, winner_id, w_slug, confidence)
        if result == "written":
            print(f"  WROTE {md_path.name} (e{loser_id} -> e{winner_id})")
            written_count += 1
        elif result == "already_applied":
            print(f"  NO-OP {md_path.name}: already merged into e{winner_id}")
        elif result == "already_merged_elsewhere":
            print(
                f"  SKIP {md_path.name}: already merged into a DIFFERENT winner "
                f"(leaving unchanged — run graph-build.py to re-check)",
                file=sys.stderr,
            )
        else:
            print(f"  WARN {md_path.name}: unexpected result '{result}'", file=sys.stderr)

    print(f"\n  {written_count} file(s) written.")
    print(
        "  Next: run graph-export.py then graph-build.py to rebuild the derived db."
    )


if __name__ == "__main__":
    main()
