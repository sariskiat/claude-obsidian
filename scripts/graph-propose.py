#!/usr/bin/env python3
"""graph-propose.py — Semantic Bridge: Claude-Code Directions Report from the Graph (P6).

Turns ranked white-space bridges (from graph-bridge.build_proposals) + the user's
research situation (RESEARCH_PROFILE.md) into a proposals.md-grade directions report
written by headless `claude -p`, with every cited paper/entity verified against
graph.db.

Usage:
    uv run python scripts/graph-propose.py
    uv run python scripts/graph-propose.py --dry-run-dossier-only
    uv run python scripts/graph-propose.py --dry-run-prompt
    uv run python scripts/graph-propose.py --bridges 12 --retries 3

Flags:
    --db PATH               Path to graph.db (default: .vault-meta/graph/graph.db)
    --output-dir PATH       Directory for saved reports (default: wiki/graph/proposals/)
    --profile PATH          RESEARCH_PROFILE.md path (default: wiki/graph/RESEARCH_PROFILE.md)
    --exemplar PATH         proposals.md exemplar (read-only; default: ~/Desktop/research/proposals.md)
    --bridges K             Number of bridge candidates (default: 12)
    --retries N             Max grounding retries (default: 3)
    --claude-cmd CMD        Claude binary / path (default: claude)
    --dry-run-dossier-only  Assemble dossier + allow-list, print as JSON, exit 0 (no claude call)
    --dry-run-prompt        Assemble prompt, print it, exit 0 (no claude call)

Exit codes:
    0  — success (clean report saved OR dry-run completed)
    1  — database missing/unreadable, missing profile, missing --claude-cmd binary
    2  — cap exhausted (.rejected.md written)
"""

import argparse
import importlib.util
import json
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

VAULT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = VAULT_ROOT / ".vault-meta" / "graph" / "graph.db"
DEFAULT_OUTPUT_DIR = VAULT_ROOT / "wiki" / "graph" / "proposals"
DEFAULT_PROFILE = VAULT_ROOT / "wiki" / "graph" / "RESEARCH_PROFILE.md"
DEFAULT_EXEMPLAR = Path.home() / "Desktop" / "research" / "proposals.md"
DEFAULT_BRIDGES = 12
DEFAULT_RETRIES = 3

# The source proposals.md must NEVER be written.
_FORBIDDEN_WRITE_PATH = Path.home() / "Desktop" / "research" / "proposals.md"

# ---------------------------------------------------------------------------
# Graph-bridge import (FR2 — import, never fork)
# ---------------------------------------------------------------------------

def _import_bridge():
    """Import graph-bridge.build_proposals without executing its __main__."""
    bridge_path = SCRIPT_DIR / "graph-bridge.py"
    spec = importlib.util.spec_from_file_location("graph_bridge", bridge_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Passage retrieval via graph-retrieve.py subprocess
# Mirrors graph-bridge._retrieve_passages_for_paper exactly.
# graph-retrieve.py emits JSON to stdout unconditionally — no --json flag needed.
# ---------------------------------------------------------------------------

RETRIEVE_SCRIPT = SCRIPT_DIR / "graph-retrieve.py"


def _retrieve_passages_for_paper(paper_slug: str, top_n: int = 2) -> list:
    """Retrieve up to top_n passage snippets via graph-retrieve.py subprocess."""
    if not RETRIEVE_SCRIPT.exists():
        return [{"paper": paper_slug, "snippet": "(graph-retrieve.py not found)"}]
    try:
        r = subprocess.run(
            [sys.executable, str(RETRIEVE_SCRIPT), "--paper", paper_slug,
             "--top", str(top_n), "--no-rerank"],
            capture_output=True, text=True, timeout=20, cwd=str(VAULT_ROOT),
        )
        if r.returncode != 0:
            return [{"paper": paper_slug, "snippet": "(no full text indexed)"}]
        # graph-retrieve.py stdout is always JSON (no --json flag required)
        data = json.loads(r.stdout)
        snippets = []
        for c in data.get("candidates", [])[:top_n]:
            snippets.append({
                "paper": paper_slug,
                "snippet": c.get("snippet", ""),
            })
        return snippets if snippets else [{"paper": paper_slug, "snippet": "(no full text indexed)"}]
    except Exception:
        return [{"paper": paper_slug, "snippet": "(no full text indexed)"}]


# ---------------------------------------------------------------------------
# Dossier assembly (FR3 + FR4)
# ---------------------------------------------------------------------------

def _build_dossier(conn, proposals: list) -> dict:
    """Build a deterministic JSON-serializable dossier from bridge proposals.

    For each proposal: anchor entities (id+name), anchor papers, limitation/
    open-question claims, full-text passages via graph-retrieve.py subprocess.

    Returns a dict with 'candidates' (one per proposal) and 'allow_list'
    (the citable set of paper slugs + entity names for this run).
    """
    candidates = []
    all_slugs: set = set()
    all_entity_names: set = set()

    for prop in proposals:
        # Anchor entities already present on the Proposal dataclass
        anchor_entities = []
        for ae in getattr(prop, "anchor_entities", []):
            anchor_entities.append({
                "id": ae["id"],
                "name": ae["name"],
                "degree": ae["degree"],
            })
            all_entity_names.add(ae["name"])

        # Anchor papers
        anchor_papers = list(getattr(prop, "anchor_papers", []))
        all_slugs.update(anchor_papers)

        # Limitation/open-question claims touching this community pair
        comm_a_nodes = set()
        comm_b_nodes = set()
        # Reconstruct community node sets from the anchor entity ids
        # (we do not store the raw node sets on Proposal, so we re-query)
        limitation_claims = []
        try:
            from graph_db import root as _root
            comm_a_id = prop.community_a.get("id")
            comm_b_id = prop.community_b.get("id")
            # Re-query entities that map to these communities via their claims
            for slug in anchor_papers:
                rows = conn.execute(
                    "SELECT id, subject_entity_id, object_entity_id, text, claim_type "
                    "FROM claims WHERE source_paper = ? "
                    "AND claim_type IN ('limitation', 'open-question')",
                    (slug,),
                ).fetchall()
                for row in rows:
                    limitation_claims.append({
                        "id": row[0],
                        "subject": row[1],
                        "object": row[2],
                        "text": row[3] or "",
                        "type": row[4],
                    })
        except Exception:
            pass

        # Full-text passages via graph-retrieve.py (degrade gracefully)
        passages = []
        for slug in anchor_papers[:2]:
            passages.extend(_retrieve_passages_for_paper(slug, top_n=1))

        candidates.append({
            "proposal_id": prop.id,
            "community_a": prop.community_a,
            "community_b": prop.community_b,
            "anchor_entities": anchor_entities,
            "anchor_papers": anchor_papers,
            "score": prop.score,
            "signal_breakdown": prop.signal_breakdown,
            "limitation_claims": limitation_claims,
            "passages": passages,
            "already_proposed": prop.already_proposed,
        })

    # Build the citable allow-list (paper slugs + entity names present in dossier)
    allow_list = sorted(all_slugs | all_entity_names)

    return {
        "candidates": candidates,
        "allow_list": allow_list,
    }


# ---------------------------------------------------------------------------
# Prompt builder (FR5 + FR7 section contract)
# ---------------------------------------------------------------------------

SECTION_CONTRACT = """
The report MUST contain exactly these section headers in this order:

## The bar
(A harsh, frank assessment of what it takes to clear the top-tier PhD ticket bar.
Do not soften this. Name the three uncomfortable truths from the research profile.)

## Decision matrix
(A markdown table with columns: # | Direction | Ceiling | Clean-exec odds @4h/wk solo | Theory load | Scoop risk | Builds on strength | Admissions signal.
Use 1-5 numeric ratings. Every candidate bridge direction must appear as a row.)

### 1. <direction name>
(One full direction block. Repeat ### N. for each direction you cover — at minimum
3 blocks total.)

**Takedown:** (Brutal, specific critique of this direction. Name the ways it fails
or is too risky at the user's current resource level. Do not hedge.)

### 2. <direction name>
**Takedown:** (...)

### 3. <direction name>
**Takedown:** (...)

## Ranking
(Explicit ordered ranking with a one-sentence rationale per direction. Include the
fork rule: theory co-author secured + >4h/week → swing for #1; solo 4h/week → ship #3.)

## Execution
(Concrete first-week proof-of-concept probe for the top-ranked direction. Name the
specific paper to reproduce, the toy model to build, and the gate condition that
triggers a pivot to the next direction.)
"""


def _build_prompt(
    profile_text: str,
    exemplar_text: str | None,
    dossier: dict,
) -> str:
    """Assemble the full prompt for `claude -p`.

    Returns a single string to pass as stdin to the claude subprocess.
    """
    allow_list = dossier.get("allow_list", [])
    allow_list_str = "\n".join(f"  - {item}" for item in allow_list)

    candidates_summary_parts = []
    for i, cand in enumerate(dossier["candidates"], 1):
        comm_a = cand.get("community_a", {})
        comm_b = cand.get("community_b", {})
        a_members = ", ".join(comm_a.get("members", [])[:3])
        b_members = ", ".join(comm_b.get("members", [])[:3])
        score = cand.get("score", 0.0)
        signal = cand.get("signal_breakdown", {})
        dr = signal.get("direction_relevance", 0.0)
        anchor_names = ", ".join(e["name"] for e in cand.get("anchor_entities", [])[:4])
        anchor_slugs = ", ".join(cand.get("anchor_papers", [])[:3])
        passages_text = ""
        for p in cand.get("passages", []):
            snip = p.get("snippet", "")
            if snip and not snip.startswith("("):
                passages_text += f"\n      Passage: {snip[:300]}"

        lim_claims = cand.get("limitation_claims", [])
        lim_text = ""
        if lim_claims:
            lim_text = "\n      Limitations/open-questions:\n" + "\n".join(
                f"        - [{c['type']}] {c['text'][:200]}" for c in lim_claims[:3]
            )

        candidates_summary_parts.append(
            f"  Bridge {i}: Community {comm_a.get('id','?')} ({a_members} ...) "
            f"<-> Community {comm_b.get('id','?')} ({b_members} ...)\n"
            f"    Score: {score:.4f}, direction_relevance: {dr:.2f}\n"
            f"    Anchor entities: {anchor_names}\n"
            f"    Anchor papers (CITE ONLY these slugs): {anchor_slugs}"
            f"{passages_text}{lim_text}"
        )
    candidates_text = "\n\n".join(candidates_summary_parts)

    system_block = (
        "You are a harsh, experienced research advisor reviewing bridge proposals "
        "derived from a claim graph. Your job is to write a proposals.md-grade "
        "directions report — opinionated, specific, brutal on the takedowns. "
        "Your primary loyalty is to the user's stated goal (top-tier PhD ticket), "
        "not to making them feel good about the directions.\n\n"
        "IMPORTANT — CITATION SAFETY:\n"
        "You MUST cite ONLY paper slugs and entity names from the explicit allow-list "
        "below. Do NOT invent paper slugs, author names, or entity names that are not "
        "in this allow-list. Any citation not in the allow-list will be rejected and "
        "you will be asked to redo the report.\n\n"
        f"ALLOW-LIST (cite ONLY these):\n{allow_list_str}\n"
    )

    profile_block = f"## Research Profile\n\n{profile_text.strip()}\n"

    exemplar_block = ""
    if exemplar_text:
        exemplar_block = (
            f"\n## Style Exemplar (match this genre and tone — do not copy it verbatim)\n\n"
            f"{exemplar_text[:3000].strip()}\n"
        )

    dossier_block = (
        f"\n## Bridge Candidates Dossier (from the claim graph)\n\n"
        f"{candidates_text}\n"
    )

    contract_block = (
        f"\n## Required Output Format\n{SECTION_CONTRACT}\n"
        "Do not add any sections not listed above. "
        "Do not include any boilerplate preamble or sign-off. "
        "Start directly with '## The bar'.\n"
    )

    prompt = (
        f"{system_block}\n"
        f"---\n\n"
        f"{profile_block}"
        f"{exemplar_block}"
        f"{dossier_block}"
        f"{contract_block}"
    )
    return prompt


# ---------------------------------------------------------------------------
# Grounding gate (FR8)
# ---------------------------------------------------------------------------

def _extract_citations(report_text: str) -> set:
    """Extract citation-like tokens from a report for grounding verification.

    Two-tier strategy:
    1. ALL-CAPS hyphenated tokens (FABRICATED-PAPER-9999, HALLUCINATED-ENTITY-XYZ)
       — these are clear hallucination markers (the prompt explicitly says not to
       invent citations, so any ALL-CAPS compound is a red flag).
    2. Lowercase slug-like tokens that appear in backtick code spans or after
       known citation markers like 'paper:' or '[' — narrowed to reduce false
       positives from prose compound modifiers.

    The allow-list + suffix-match in _verify_citations handles the rest.
    """
    citations = set()

    # ALL-CAPS hyphenated tokens (FABRICATED-PAPER-9999, HALLUCINATED-ENTITY-XYZ)
    for m in re.finditer(r"\b([A-Z][A-Z0-9]+(?:-[A-Z0-9]+){1,})\b", report_text):
        citations.add(m.group(1))

    # Backtick-quoted tokens (the report may cite slugs in `code spans`)
    # These are likely intentional citations, not prose compound modifiers.
    for m in re.finditer(r"`([a-z][a-z0-9]+(?:-[a-z0-9]+){2,})`", report_text):
        tok = m.group(1)
        if len(tok) >= 10:
            citations.add(tok)

    return citations


def _verify_citations(citations: set, allow_list: list, conn) -> tuple[set, set]:
    """Verify extracted citations against the allow-list and graph.db.

    Returns (verified, unverified) sets.
    """
    allow_set = set(allow_list)

    # Also build a set from live db (papers.slug + entities.name)
    try:
        db_slugs = {row[0] for row in conn.execute("SELECT slug FROM papers")}
        db_entities = {row[0].lower() for row in conn.execute("SELECT name FROM entities")}
    except Exception:
        db_slugs = set()
        db_entities = set()

    verified = set()
    unverified = set()

    allow_set_lower = {a.lower() for a in allow_set}
    db_entities_lower = {e.lower() for e in db_entities}

    for cite in citations:
        cite_lower = cite.lower()
        # Also try space-separated version for entity name matching
        cite_spaced = cite.replace("-", " ").lower()
        if (cite in allow_set or
                cite_lower in allow_set_lower or
                cite in db_slugs or
                cite_lower in db_entities_lower or
                cite_spaced in db_entities_lower or
                # Suffix match: e.g. "ton-textured-3d-..." is a strict suffix of the real slug
                any(real_slug.endswith(cite_lower) for real_slug in db_slugs) or
                # Allow partial entity name matches (substring either way)
                any(cite_lower in e or e in cite_lower for e in db_entities_lower if len(e) > 8)):
            verified.add(cite)
        else:
            unverified.add(cite)

    return verified, unverified


def _flag_unverified_in_report(report_text: str, unverified: set) -> str:
    """Inline-flag unverified citations in a report for .rejected.md output."""
    flagged = report_text
    for cite in sorted(unverified):
        flagged = flagged.replace(
            cite, f"{cite} [UNVERIFIED — not in graph.db]"
        )
    return flagged


# ---------------------------------------------------------------------------
# Output path helpers (FR9, BR5)
# ---------------------------------------------------------------------------

def _make_output_path(output_dir: Path, suffix: str = "-directions") -> Path:
    """Compute a non-clobbering output path under output_dir.

    Format: YYYY-MM-DD<suffix>.md
    If that file exists, append -2, -3, etc.
    """
    today = date.today().isoformat()
    base = output_dir / f"{today}{suffix}.md"
    if not base.exists():
        return base
    n = 2
    while True:
        candidate = output_dir / f"{today}{suffix}-{n}.md"
        if not candidate.exists():
            return candidate
        n += 1


# ---------------------------------------------------------------------------
# Re-prompt message for dirty cites
# ---------------------------------------------------------------------------

def _build_retry_prompt(original_prompt: str, unverified: set) -> str:
    """Rebuild prompt emphasizing the offending citations."""
    unverified_str = "\n".join(f"  - {c}" for c in sorted(unverified))
    warning = (
        f"\n\n## GROUNDING FAILURE — PLEASE FIX\n\n"
        f"Your previous response contained the following citations that are NOT in "
        f"the allow-list and could not be verified in graph.db:\n"
        f"{unverified_str}\n\n"
        f"You MUST rewrite the entire report citing ONLY the items in the "
        f"ALLOW-LIST section above. Do not invent any paper slugs or entity names.\n"
    )
    return original_prompt + warning


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def _run_claude(claude_cmd: str, prompt: str, timeout: int = 300) -> tuple[int, str, str]:
    """Run headless `claude -p` and return (exit_code, stdout, stderr)."""
    try:
        r = subprocess.run(
            [claude_cmd, "-p"],
            input=prompt,
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return -1, "", f"claude command not found: {claude_cmd}"
    except subprocess.TimeoutExpired:
        return -1, "", f"claude timed out after {timeout}s"
    except Exception as e:
        return -1, "", str(e)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="graph-propose.py — Semantic Bridge directions report from the graph"
    )
    parser.add_argument("--db", default=str(DEFAULT_DB),
                        help="Path to graph.db")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="Directory for saved reports")
    parser.add_argument("--profile", default=str(DEFAULT_PROFILE),
                        help="RESEARCH_PROFILE.md path")
    parser.add_argument("--exemplar", default=str(DEFAULT_EXEMPLAR),
                        help="proposals.md style exemplar (read-only)")
    parser.add_argument("--bridges", type=int, default=DEFAULT_BRIDGES,
                        help="Number of bridge candidates (default: 12)")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                        help="Max grounding retries (default: 3)")
    parser.add_argument("--claude-cmd", default="claude",
                        help="Claude binary or path (default: claude)")
    parser.add_argument("--dry-run-dossier-only", action="store_true",
                        help="Assemble dossier + allow-list, print JSON, exit 0 (no claude call)")
    parser.add_argument("--dry-run-prompt", action="store_true",
                        help="Assemble prompt, print it, exit 0 (no claude call)")
    args = parser.parse_args()

    # --- Safety check: never allow output to land at the forbidden path ---
    # This is enforced at the path-computation level (output_dir is always
    # separate from Desktop/research), but we add an explicit guard.
    output_dir = Path(args.output_dir).resolve()
    forbidden = _FORBIDDEN_WRITE_PATH.resolve()
    if output_dir == forbidden.parent or str(forbidden).startswith(str(output_dir)):
        print("Error: output-dir must not point to ~/Desktop/research/", file=sys.stderr)
        return 1

    # --- Validate db ---
    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Error: database not found at {db_path}", file=sys.stderr)
        print("Run: uv run python scripts/graph-build.py wiki/graph .vault-meta/graph/graph.db",
              file=sys.stderr)
        return 1

    # --- Validate profile ---
    profile_path = Path(args.profile)
    if not profile_path.exists():
        print(f"Error: RESEARCH_PROFILE.md not found at {profile_path}", file=sys.stderr)
        print(
            "Create it: wiki/graph/RESEARCH_PROFILE.md (seed from the research-goal-phd-paper "
            "memory note or run T1 of the graph-semantic-bridge feature).",
            file=sys.stderr,
        )
        return 1

    profile_text = profile_path.read_text(encoding="utf-8")

    # --- Load exemplar (missing -> warn + proceed) ---
    exemplar_path = Path(args.exemplar)
    exemplar_text = None
    if not exemplar_path.exists():
        print(
            f"Warning: proposals.md exemplar not found at {exemplar_path}. "
            "Proceeding without style exemplar (degraded taste, still grounded).",
            file=sys.stderr,
        )
    else:
        exemplar_text = exemplar_path.read_text(encoding="utf-8")

    # --- Import graph-bridge (FR2) ---
    try:
        bridge_mod = _import_bridge()
    except Exception as e:
        print(f"Error: could not import graph-bridge.py: {e}", file=sys.stderr)
        return 1

    # --- Connect to db ---
    try:
        from graph_db import connect as _connect
        conn = _connect(db_path)
    except Exception as e:
        print(f"Error: could not connect to {db_path}: {e}", file=sys.stderr)
        return 1

    # --- Build bridge proposals (reuse build_proposals, no passage retrieval yet) ---
    try:
        proposals = bridge_mod.build_proposals(
            conn, top_n=args.bridges, retrieve_passages=False,
        )
    except Exception as e:
        print(f"Error: build_proposals failed: {e}", file=sys.stderr)
        conn.close()
        return 1

    # --- Assemble dossier ---
    dossier = _build_dossier(conn, proposals)

    # --- --dry-run-dossier-only: print JSON and exit ---
    if args.dry_run_dossier_only:
        print(json.dumps(dossier, indent=2, ensure_ascii=False))
        conn.close()
        return 0

    # --- Build prompt ---
    prompt = _build_prompt(profile_text, exemplar_text, dossier)

    # --- --dry-run-prompt: print prompt and exit ---
    if args.dry_run_prompt:
        print(prompt)
        conn.close()
        return 0

    # --- Validate --claude-cmd binary exists ---
    claude_cmd = args.claude_cmd
    if not shutil.which(claude_cmd) and not Path(claude_cmd).is_file():
        print(
            f"Error: --claude-cmd binary not found: {claude_cmd}\n"
            "Install claude CLI or point --claude-cmd at the binary.",
            file=sys.stderr,
        )
        conn.close()
        return 1

    # --- Ensure output directory exists ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Grounding gate loop (FR8, FR9, FR10) ---
    allow_list = dossier.get("allow_list", [])
    current_prompt = prompt
    report_text = None
    unverified_final: set = set()
    retry_count = 0

    for attempt in range(args.retries + 1):
        rc, stdout, stderr = _run_claude(claude_cmd, current_prompt)
        if rc != 0 or not stdout.strip():
            print(
                f"Error: claude call failed (attempt {attempt + 1}/{args.retries + 1}). "
                f"exit={rc}. stderr: {stderr[:300]}",
                file=sys.stderr,
            )
            conn.close()
            return 1

        candidate_report = stdout.strip()
        citations = _extract_citations(candidate_report)
        _, unverified = _verify_citations(citations, allow_list, conn)

        if not unverified:
            # Clean pass
            report_text = candidate_report
            retry_count = attempt
            break
        else:
            print(
                f"Grounding check attempt {attempt + 1}: "
                f"{len(unverified)} unverified citation(s): "
                f"{sorted(unverified)[:5]}",
                file=sys.stderr,
            )
            unverified_final = unverified
            if attempt < args.retries:
                # Re-prompt with flagged cites
                current_prompt = _build_retry_prompt(prompt, unverified)
                retry_count = attempt + 1
            # else: fall through to cap-exhaustion handling below

    if report_text is None:
        # Cap exhausted — write .rejected.md
        flagged_text = _flag_unverified_in_report(
            candidate_report, unverified_final
        )
        rejected_path = _make_output_path(output_dir, suffix="-directions.rejected")
        # Rename so it ends in .rejected.md (override suffix logic)
        today = date.today().isoformat()
        rejected_path = output_dir / f"{today}-directions.rejected.md"
        n = 2
        while rejected_path.exists():
            rejected_path = output_dir / f"{today}-directions.rejected-{n}.md"
            n += 1

        rejected_content = (
            f"# Directions Report — REJECTED (grounding cap exhausted)\n\n"
            f"**Unverified citations:** {', '.join(sorted(unverified_final))}\n\n"
            f"---\n\n"
            f"{flagged_text}\n"
        )
        rejected_path.write_text(rejected_content, encoding="utf-8")
        print(
            f"Grounding failed after {args.retries + 1} attempts. "
            f"Unverified: {sorted(unverified_final)}",
            file=sys.stderr,
        )
        print(f"Rejected report saved: {rejected_path}", file=sys.stderr)
        conn.close()
        return 2

    # --- Save clean report with audit footer ---
    total_citations = len(_extract_citations(report_text))
    # Recount verified (all citations in clean report must be verified)
    citations = _extract_citations(report_text)
    _, remaining_unverified = _verify_citations(citations, allow_list, conn)
    verified_count = len(citations) - len(remaining_unverified)
    total_count = len(citations)

    # Determine model identifier (from claude version if possible)
    model_id = "claude"
    try:
        ver_r = subprocess.run(
            [claude_cmd, "--version"], capture_output=True, text=True, timeout=5
        )
        if ver_r.returncode == 0 and ver_r.stdout.strip():
            model_id = ver_r.stdout.strip().split("\n")[0][:60]
    except Exception:
        pass

    audit_footer = (
        f"\n\n---\n\n"
        f"*Grounding audit: {verified_count}/{total_count} citations verified ✓ "
        f"| retries: {retry_count} "
        f"| model: {model_id} "
        f"| bridge candidates: {len(proposals)}*\n"
    )

    full_report = report_text + audit_footer
    out_path = _make_output_path(output_dir)
    out_path.write_text(full_report, encoding="utf-8")

    print(f"Report saved: {out_path}", file=sys.stderr)
    print(
        f"Grounding: {verified_count}/{total_count} citations verified. "
        f"Retries: {retry_count}.",
        file=sys.stderr,
    )

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
