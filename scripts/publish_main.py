#!/usr/bin/env -S uv run --no-project --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = []
# ///
"""Publish committed dev changes through the existing, filtered main worktree."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
from dataclasses import dataclass


PAPER = "Parametric-Adaptation-Methods-RAG.pdf"
UNI_PAPER = "Parametric-Adaptation-Methods-RAG_uni.pdf"
PUBLIC_DOCS = {"README.md", "experiments/README.md"}
PRIVATE_ROOTS = {
    ".git", ".claude", ".codex", ".idea", ".vscode", ".old", ".venv",
    "docs", "external", "logs", "memory_bank", "output", "tmp", "models",
    "trained_d2l", "term-paper", "term-paper_2", "term-paper_3", "scripts", "tests",
}


class PublicationError(Exception):
    """A failed safety check; never repair it by resetting a worktree."""


@dataclass(frozen=True)
class Entry:
    mode: str
    oid: str


def run_git(root: Path, *args: str, data: bytes | None = None,
            env: dict[str, str] | None = None,
            accepted: tuple[int, ...] = (0,)) -> subprocess.CompletedProcess[bytes]:
    process_env = os.environ.copy()
    for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        process_env.pop(key, None)
    process_env.update(GIT_TERMINAL_PROMPT="0", GIT_MERGE_AUTOEDIT="no")
    process_env.update(env or {})
    result = subprocess.run(
        ["git", "--no-pager", "-c", "color.ui=false", *args], cwd=root,
        input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=process_env,
    )
    if result.returncode not in accepted:
        detail = result.stderr.decode(errors="replace").strip()
        raise PublicationError(f"git {args[0]} failed: {detail or result.returncode}")
    return result


def git(root: Path, *args: str, **kwargs) -> bytes:
    return run_git(root, *args, **kwargs).stdout


def revision(root: Path, ref: str = "HEAD") -> str:
    return git(root, "rev-parse", "--verify", ref).decode().strip()


def ancestor(root: Path, before: str, after: str) -> bool:
    return run_git(root, "merge-base", "--is-ancestor", before, after,
                   accepted=(0, 1)).returncode == 0


def worktrees(root: Path) -> dict[str, Path]:
    found = {}
    path = None
    for field in git(root, "worktree", "list", "--porcelain", "-z").split(b"\0"):
        if field.startswith(b"worktree "):
            path = Path(os.fsdecode(field[9:])).resolve()
        elif field.startswith(b"branch refs/heads/") and path is not None:
            found[field[18:].decode()] = path
    if not {"dev", "main"} <= found.keys():
        raise PublicationError("Both dev and main must already have separate worktrees.")
    return found


def check_worktree(root: Path, branch: str, *, preview: bool = False) -> None:
    if git(root, "branch", "--show-current").decode().strip() != branch:
        raise PublicationError(f"Expected {branch} in {root}.")
    for state in ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply"):
        path = Path(os.fsdecode(git(root, "rev-parse", "--path-format=absolute",
                                    "--git-path", state).strip()))
        if path.exists():
            raise PublicationError(f"Finish or abort the active Git operation in {branch} first.")
    if any(record and record[:1] != b"H"
           for record in git(root, "ls-files", "-v", "-z").split(b"\0")):
        raise PublicationError(f"Clear assume-unchanged/skip-worktree or unresolved index flags in {branch} first.")
    if git(root, "status", "--porcelain=v1", "--untracked-files=all", "-z"):
        if preview and branch == "dev":
            print("Dry run uses committed dev only; uncommitted changes are NOT included.")
        else:
            raise PublicationError(f"{branch} is dirty. Commit intended changes first; nothing was stashed.")


def tree_entries(root: Path, ref: str) -> dict[str, Entry]:
    entries = {}
    for record in git(root, "ls-tree", "-r", "-z", ref).split(b"\0"):
        if record:
            metadata, name = record.split(b"\t", 1)
            mode, _, oid = metadata.decode().split()
            entries[os.fsdecode(name)] = Entry(mode, oid)
    return entries


def private_path(name: str) -> bool:
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or str(path) != name or ".." in path.parts
            or any(ord(char) < 32 for char in name) or "\\" in name):
        return True
    if path.parts[0] in PRIVATE_ROOTS:
        return True
    if any(part in {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
           for part in path.parts):
        return True
    if path.name in {"AGENTS.md", "CLAUDE.md", ".DS_Store", ".lock"}:
        return True
    if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
        return True
    if (name in {UNI_PAPER, "Parametric-Adaptation-Methods-RAG_frontpage_uni.pdf"}
            or path.name.startswith("Eigenstaendigkeitserklaerung_")):
        return True
    if path.suffix in {".pyc", ".pyo", ".log", ".pt", ".bin", ".safetensors"}:
        return True
    return (name.startswith("results/figures/")
            or (path.parts[0] == "results"
                and (any(part in {"index", "faiss_index", "smoke"} for part in path.parts)
                     or path.name.endswith("_progress.json"))))


def ignored_paths(root: Path, paths: set[str]) -> set[str]:
    if not paths:
        return set()
    result = run_git(root, "check-ignore", "--no-index", "-z", "--stdin",
                     data=b"\0".join(os.fsencode(p) for p in sorted(paths)) + b"\0",
                     accepted=(0, 1))
    return {os.fsdecode(p) for p in result.stdout.split(b"\0") if p}


def remote_refs(root: Path) -> dict[str, str]:
    fetch_urls = git(root, "remote", "get-url", "--all", "origin").splitlines()
    push_urls = git(root, "remote", "get-url", "--push", "--all", "origin").splitlines()
    if len(fetch_urls) != 1 or fetch_urls != push_urls:
        raise PublicationError("This command requires one origin URL, shared for fetch and push.")
    refs = {}
    for line in git(root, "ls-remote", "--refs", "origin",
                    "refs/heads/dev", "refs/heads/main").splitlines():
        oid, ref = line.decode().split()
        refs[ref.removeprefix("refs/heads/")] = oid
    if refs.keys() != {"dev", "main"}:
        raise PublicationError("origin must already contain both dev and main.")
    return refs


def published_source(root: Path, main: str, source: str) -> str:
    line = git(root, "log", "--first-parent", "--min-parents=2", "-n", "1",
               "--format=%P", main).decode().strip().split()
    if len(line) != 2 or not ancestor(root, line[1], source):
        raise PublicationError("Cannot identify the last dev publication from main's merge parents.")
    return line[1]


def merge_document(root: Path, directory: Path, name: str,
                   main: Entry, base: Entry, source: Entry) -> Entry:
    if {main.mode, base.mode, source.mode} != {"100644"}:
        raise PublicationError(f"Resolve the publication-specific mode change manually: {name}")
    directory.mkdir()
    files = []
    for label, entry in (("main", main), ("base", base), ("dev", source)):
        path = directory / label
        path.write_bytes(git(root, "cat-file", "blob", entry.oid))
        files.append(str(path))
    result = run_git(root, "merge-file", "--stdout", "-L", "main", "-L", "published dev",
                     "-L", "dev", *files, accepted=(0, 1))
    if result.returncode:
        raise PublicationError(f"Conflicting public/dev edits in {name}; main was not changed.")
    oid = git(root, "hash-object", "-w", "--stdin", data=result.stdout).decode().strip()
    return Entry("100644", oid)


def build_public_tree(root: Path, source: str, main: str, base: str,
                      args: argparse.Namespace, scratch: Path) -> tuple[str, list[str]]:
    current, original, incoming = (tree_entries(root, ref) for ref in (main, base, source))
    include, skip = set(args.include), set(args.skip)
    ignored = ignored_paths(root, set(current) | set(incoming))
    if include & skip:
        raise PublicationError("A path cannot be both --include and --skip.")
    for name in set(current) | include:
        entry = incoming.get(name) if name in include else current[name]
        if (private_path(name) or name in ignored or entry is None
                or entry.mode not in {"100644", "100755"}):
            raise PublicationError(f"Not an eligible public regular file: {name}")
    additions = {name for name in incoming.keys() - original.keys() - current.keys()
                 if not private_path(name) and name not in ignored}
    if skip - additions:
        raise PublicationError("--skip only accepts newly added public candidates: "
                               + ", ".join(sorted(skip - additions)))
    unreviewed = additions - include - skip
    if unreviewed:
        raise PublicationError("Review new files; select each with --include PATH or --skip PATH:\n  "
                               + "\n  ".join(sorted(unreviewed)))

    selected = dict(current)
    for name in sorted(set(current) | include):
        if name == ".gitignore":
            continue  # main owns its publication filters.
        ours, old, theirs = current.get(name), original.get(name), incoming.get(name)
        if ours == theirs:
            continue
        if ours is not None and ours != old and name not in PUBLIC_DOCS:
            raise PublicationError(f"Bring the substantive main-only changes into dev first: {name}")
        if ours is None and name in include:
            selected[name] = theirs
        elif old == theirs:
            continue  # Keep publication-only documentation filtering.
        elif ours == old:
            if theirs is None:
                selected.pop(name, None)
            else:
                selected[name] = theirs
        elif name in PUBLIC_DOCS and ours and old and theirs:
            selected[name] = merge_document(root, scratch / f"merge-{name.replace('/', '-')}",
                                             name, ours, old, theirs)
        else:
            raise PublicationError(f"Bring the substantive main-only changes into dev first: {name}")

    changes = {name for name in current.keys() | selected.keys()
               if current.get(name) != selected.get(name)}
    for name, entry in selected.items():
        if private_path(name) or name in ignored or entry.mode not in {"100644", "100755"}:
            raise PublicationError(f"Working artifact or non-regular file in the public tree: {name}")
    if not {PAPER, "README.md", ".gitignore"} <= selected.keys():
        raise PublicationError("The primary PDF, README.md and public .gitignore must remain present.")
    readme = git(root, "cat-file", "blob", selected["README.md"].oid)
    if UNI_PAPER.encode() in readme or f"](./{PAPER})".encode() not in readme:
        raise PublicationError("The public README must link to the primary PDF only.")
    if not git(root, "cat-file", "blob", selected[PAPER].oid).startswith(b"%PDF-"):
        raise PublicationError("The primary PDF is not a PDF file.")
    for name in sorted(changes & selected.keys()):
        content = git(root, "cat-file", "blob", selected[name].oid)
        try:
            if name.endswith(".py"):
                compile(content, name, "exec")
            elif name.endswith(".json"):
                json.loads(content)
            elif name.endswith(".jsonl"):
                for line in content.splitlines():
                    if line.strip():
                        json.loads(line)
            elif name.endswith(".md") and UNI_PAPER.encode() in content:
                raise ValueError("a link to the excluded university PDF")
        except (SyntaxError, ValueError, UnicodeError) as error:
            raise PublicationError(f"Invalid public artifact {name}: {error}") from error

    index_env = {"GIT_INDEX_FILE": str(scratch / "index")}
    git(root, "read-tree", main, env=index_env)
    records = []
    for name in sorted(changes):
        entry = selected.get(name, Entry("0", "0" * len(source)))
        records.append(f"{entry.mode} {entry.oid}\t".encode() + os.fsencode(name) + b"\0")
    if records:
        git(root, "update-index", "-z", "--index-info", data=b"".join(records), env=index_env)
    tree = git(root, "write-tree", env=index_env).decode().strip()
    git(root, "diff", "--check", main, tree)
    return tree, sorted(current.keys() - selected.keys())


def check_new_paths(root: Path, main: str, tree: str) -> None:
    added = tree_entries(root, tree).keys() - tree_entries(root, main).keys()
    for name in added:
        path = root / name
        if path.exists() or path.is_symlink():
            raise PublicationError(f"A local file would be overwritten in main: {name}")
        for parent in path.parents:
            if parent == root:
                break
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise PublicationError(f"A local path blocks publication: {name}")


def save_record(common: Path, record: dict) -> None:
    path = common / "publish-main-last.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def publish(args: argparse.Namespace, roots: dict[str, Path], common: Path) -> None:
    dev, main = roots["dev"], roots["main"]
    check_worktree(dev, "dev", preview=args.dry_run)
    check_worktree(main, "main")
    source, before = revision(dev), revision(main)
    base = published_source(main, before, source)
    print(f"dev {source}\nmain before {before}\nlast published dev {base}", flush=True)
    remote = None if args.no_push or args.dry_run else remote_refs(main)
    if remote:
        for branch, local in (("dev", source), ("main", before)):
            known = run_git(main, "cat-file", "-e", f"{remote[branch]}^{{commit}}",
                            accepted=(0, 1, 128)).returncode == 0
            if not known:
                raise PublicationError(f"origin/{branch} has an unknown commit. Run git fetch origin and review it first.")
            if not ancestor(main, remote[branch], local):
                raise PublicationError(f"origin/{branch} is ahead or diverged. Fetch and reconcile it first.")
    with tempfile.TemporaryDirectory(prefix="term-paper-publish-") as directory:
        tree, deletions = build_public_tree(main, source, before, base, args, Path(directory))
    comparison = remote["main"] if remote else before
    summary = git(main, "diff", "--find-renames", "--name-status", comparison, tree).decode()
    print(summary.rstrip() if summary else "No public file changes.")
    check_new_paths(main, before, tree)
    if deletions and not args.allow_deletions:
        message = "Review these removals and pass --allow-deletions: " + ", ".join(deletions)
        if not args.dry_run:
            raise PublicationError(message)
        print(message)
    if args.dry_run:
        print("Dry run complete: no refs, worktree files or remote branches changed.")
        return

    # Check again after preparing the isolated index and immediately before changing main.
    check_worktree(dev, "dev")
    check_worktree(main, "main")
    if (revision(dev), revision(main)) != (source, before):
        raise PublicationError("A branch moved during preparation; run the command again.")
    if remote and remote_refs(main) != remote:
        raise PublicationError("Remote branches moved during preparation; main was not changed.")
    candidate = before
    record = {"dev": source, "main_before": before, "published_base": base,
              "tree": tree, "remote_before": remote}
    if tree != revision(main, f"{before}^{{tree}}"):
        candidate = git(main, "commit-tree", tree, "-p", before, "-p", source,
                        data=(args.message + "\n").encode()).decode().strip()
        parents = git(main, "show", "-s", "--format=%P", candidate).decode().split()
        message = git(main, "show", "-s", "--format=%B", candidate).decode().strip()
        if parents != [before, source] or message != args.message:
            raise PublicationError("Unexpected publication parents or commit message.")
        record.update(main_after=candidate, status="prepared")
        save_record(common, record)
        git(main, "merge", "--ff-only", "--no-edit", "--no-autostash",
            "--no-overwrite-ignore", "--no-stat", candidate)
    if revision(main) != candidate or revision(main, "HEAD^{tree}") != tree:
        raise PublicationError("main does not match the reviewed publication tree.")
    check_worktree(main, "main")
    check_worktree(dev, "dev")
    if revision(dev) != source:
        raise PublicationError("dev moved; main is local only. Inspect both branches before retrying.")
    record.update(main_after=candidate, status="local")
    save_record(common, record)
    if args.no_push:
        print(f"main {candidate}; local only (--no-push).")
        return
    if remote_refs(main) != remote:
        raise PublicationError("Remote branches moved; main is local only. Inspect them before retrying.")
    print("Pushing dev and main atomically; no force push.", flush=True)
    try:
        git(main, "push", "--atomic", "--porcelain", "--no-follow-tags", "origin",
            f"{source}:refs/heads/dev", f"{candidate}:refs/heads/main")
        if remote_refs(main) != {"dev": source, "main": candidate}:
            raise PublicationError("Remote refs do not match the expected publication.")
    except PublicationError as error:
        raise PublicationError(f"Push not verified; main is local at {candidate}. "
                               f"Fix the remote issue and rerun; no reset is needed.\n{error}") from error
    record["status"] = "pushed"
    save_record(common, record)
    if revision(dev) != source or revision(main) != candidate:
        raise PublicationError("Publication reached origin, but a local branch moved concurrently; inspect it.")
    check_worktree(dev, "dev")
    check_worktree(main, "main")
    print(f"Published main {candidate}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="preview committed snapshots without publishing")
    parser.add_argument("--no-push", action="store_true", help="create the publication locally only")
    parser.add_argument("--message", default="main update", help="one-line commit subject (default: main update)")
    parser.add_argument("--include", action="append", default=[], metavar="PATH", help="explicitly select a new public file")
    parser.add_argument("--skip", action="append", default=[], metavar="PATH", help="explicitly omit a new public candidate")
    parser.add_argument("--allow-deletions", action="store_true", help="allow reviewed deletions of existing public files")
    args = parser.parse_args()
    if not args.message.strip() or args.message != args.message.strip() or any(ord(c) < 32 for c in args.message):
        parser.error("--message must be a nonempty, single-line subject without surrounding whitespace")
    return args


def main() -> int:
    args = parse_args()
    try:
        roots = worktrees(Path.cwd())
        common = Path(os.fsdecode(git(roots["main"], "rev-parse", "--path-format=absolute",
                                      "--git-common-dir").strip()))
        with (common / "publish-main.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise PublicationError("Another publication command is already running.") from error
            publish(args, roots, common)
    except (PublicationError, OSError) as error:
        print(f"Publication stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
