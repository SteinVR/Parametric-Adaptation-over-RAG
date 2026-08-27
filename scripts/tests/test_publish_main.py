"""Integration tests use disposable worktrees and a local bare origin only."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "publish_main.py"
PAPER = "Parametric-Adaptation-Methods-RAG.pdf"
UNI = "Parametric-Adaptation-Methods-RAG_uni.pdf"
README = (
    f"# Paper\n\n[Paper](./{PAPER})\n[University version](./{UNI})\n\n"
    "## Method\n\nA fixed benchmark.\n\nRetrieval is available.\n\n"
    "The corpus stays frozen.\n\n## Results\n\nOld result summary.\n\n"
    "## Reproduction\n\nUse the committed inputs.\n\nNo training in publication.\n"
)
IGNORE = f".env\n*.pyc\n/docs/\nAGENTS.md\n/{UNI}\n/scripts/\n/results/**/smoke/\n"


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="publication-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dev = self.root / "dev с пробелами"
        self.main = self.root / "main worktree"
        self.origin = self.root / "origin.git"
        self.dev.mkdir()
        self.env = os.environ.copy()
        self.env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull,
                        GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0",
                        PYTHONDONTWRITEBYTECODE="1")
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            self.env.pop(key, None)
        self.git(self.dev, "init", "-q", "-b", "dev")
        self.git(self.dev, "config", "user.name", "Publication Test")
        self.git(self.dev, "config", "user.email", "publication@example.invalid")
        self.git(self.dev, "config", "commit.gpgSign", "false")
        for name, content in {
            PAPER: b"%PDF-1.4\noriginal fixture\n%%EOF\n",
            UNI: b"%PDF-1.4\nuniversity fixture\n%%EOF\n",
            "README.md": README,
            ".gitignore": ".env\n*.pyc\n",
            "config.py": "VALUE = 1\n",
            "src/library.py": "def value():\n    return 1\n",
            "results/metrics.json": '{"score": 0.5}\n',
            "docs/private.md": "Private working notes.\n",
            "AGENTS.md": "Private instructions.\n",
        }.items():
            self.write(self.dev, name, content)
        self.commit(self.dev, "initial dev")
        self.base = self.sha(self.dev)
        self.git(self.dev, "worktree", "add", "-q", "-b", "main", str(self.main), "dev")
        self.git(self.main, "rm", "--", "docs/private.md", "AGENTS.md")
        self.write(self.main, ".gitignore", IGNORE)
        self.write(self.main, "README.md", README + "\nPublic distribution only.\n")
        self.commit(self.main, "public base")
        before = self.sha(self.main)
        tree = self.git(self.main, "rev-parse", "HEAD^{tree}").strip()
        merge = self.git(self.main, "commit-tree", tree, "-p", before, "-p", self.base,
                         data=b"main update\n").strip()
        self.git(self.main, "merge", "--ff-only", merge)
        self.git(self.main, "rm", "--", UNI)
        self.write(self.main, "README.md", (self.main / "README.md").read_text().replace(
            f"[University version](./{UNI})\n", ""))
        self.commit(self.main, "remove university pdf")
        self.git(self.root, "init", "-q", "--bare", str(self.origin))
        self.git(self.dev, "remote", "add", "origin", str(self.origin))
        self.git(self.dev, "push", "-q", "-u", "origin", "dev", "main")
        self.before = self.sha(self.main)

    def git(self, cwd, *args, data=None):
        result = subprocess.run(["git", "--no-pager", *args], cwd=cwd, env=self.env,
                                input=data, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        return result.stdout.decode()

    def sha(self, root, ref="HEAD"):
        return self.git(root, "rev-parse", ref).strip()

    def write(self, root, name, content):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content.encode() if isinstance(content, str) else content)

    def commit(self, root, subject="dev change"):
        self.git(root, "add", "-A")
        self.git(root, "commit", "-q", "-m", subject)
        return self.sha(root)

    def invoke(self, *args, expected=0, cwd=None):
        result = subprocess.run([sys.executable, str(SCRIPT), *args], cwd=cwd or self.dev,
                                env=self.env, capture_output=True, text=True)
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, expected, output)
        return output

    def change_config(self):
        self.write(self.dev, "config.py", "VALUE = 2\n")
        return self.commit(self.dev)

    def assert_clean(self):
        for root in (self.dev, self.main):
            self.assertEqual(self.git(root, "status", "--porcelain=v1"), "")

    def test_publish_preserves_public_filtering_and_has_two_parents(self):
        self.write(self.dev, "config.py", "VALUE = 2\n")
        self.write(self.dev, "README.md", README.replace("Old result summary.", "Updated result summary."))
        self.write(self.dev, PAPER, b"%PDF-1.4\nupdated fixture\n%%EOF\n")
        self.write(self.dev, "docs/private.md", "Updated private notes.\n")
        source = self.commit(self.dev)
        self.invoke()
        head = self.sha(self.main)
        self.assertEqual(self.git(self.main, "show", "-s", "--format=%P", "HEAD").strip(),
                         f"{self.before} {source}")
        self.assertEqual(self.git(self.main, "show", "-s", "--format=%B", "HEAD").strip(), "main update")
        self.assertEqual(self.git(self.main, "show", "-s", "--format=%b", "HEAD").strip(), "")
        readme = (self.main / "README.md").read_text()
        self.assertIn("Updated result summary.", readme)
        self.assertIn("Public distribution only.", readme)
        self.assertNotIn(UNI, readme)
        self.assertEqual((self.main / ".gitignore").read_text(), IGNORE)
        self.assertEqual((self.main / PAPER).read_bytes(), (self.dev / PAPER).read_bytes())
        self.assertFalse((self.main / UNI).exists())
        self.assertFalse((self.main / "docs/private.md").exists())
        self.assertTrue((self.dev / UNI).exists())
        self.assertEqual(self.sha(self.dev), source)
        self.assertEqual(self.sha(self.origin, "main"), head)
        self.assertEqual(self.sha(self.origin, "dev"), source)
        self.assert_clean()
        self.invoke()
        self.assertEqual(self.sha(self.main), head, "a repeat must not create another commit")

    def test_dirty_dev_is_rejected_but_preview_reads_only_committed_files(self):
        self.change_config()
        self.write(self.dev, "config.py", "VALUE = 999\n")
        self.assertIn("dev is dirty", self.invoke("--no-push", expected=1))
        self.assertIn("uncommitted changes are NOT included", self.invoke("--dry-run"))
        self.assertEqual(self.sha(self.main), self.before)
        self.assertEqual((self.main / "config.py").read_text(), "VALUE = 1\n")
        self.assertEqual((self.dev / "config.py").read_text(), "VALUE = 999\n")

    def test_staged_dev_changes_are_not_commits(self):
        self.write(self.dev, "config.py", "VALUE = 2\n")
        self.git(self.dev, "add", "config.py")
        self.assertIn("dev is dirty", self.invoke(expected=1))
        self.assertEqual(self.sha(self.main), self.before)

    def test_dirty_main_is_never_overwritten(self):
        self.change_config()
        self.write(self.main, "local.txt", "Local work.\n")
        self.assertIn("main is dirty", self.invoke("--dry-run", expected=1))
        self.assertEqual((self.main / "local.txt").read_text(), "Local work.\n")
        self.assertEqual(self.sha(self.main), self.before)

    def test_hidden_index_flags_are_rejected(self):
        self.change_config()
        self.git(self.main, "update-index", "--assume-unchanged", "config.py")
        self.write(self.main, "config.py", "HIDDEN_EDIT = True\n")
        self.assertIn("assume-unchanged", self.invoke("--no-push", expected=1))
        self.assertEqual((self.main / "config.py").read_text(), "HIDDEN_EDIT = True\n")

    def test_new_file_requires_explicit_inclusion(self):
        self.write(self.dev, "src/new module.py", "NEW = True\n")
        self.commit(self.dev)
        self.assertIn("--include PATH", self.invoke("--no-push", expected=1))
        self.assertEqual(self.sha(self.main), self.before)
        self.invoke("--no-push", "--include", "src/new module.py")
        self.assertTrue((self.main / "src/new module.py").exists())
        self.assert_clean()

    def test_new_candidate_can_be_explicitly_skipped(self):
        self.write(self.dev, "local-note.txt", "Not public.\n")
        self.change_config()
        self.invoke("--no-push", "--skip", "local-note.txt")
        self.assertFalse((self.main / "local-note.txt").exists())
        self.assertEqual((self.main / "config.py").read_text(), "VALUE = 2\n")

    def test_main_only_removal_is_preserved_unless_explicitly_included(self):
        self.git(self.main, "rm", "src/library.py")
        before = self.commit(self.main, "filter optional file")
        self.invoke("--no-push")
        self.assertEqual(self.sha(self.main), before)
        self.assertFalse((self.main / "src/library.py").exists())
        self.assertTrue((self.dev / "src/library.py").exists())
        self.invoke("--no-push", "--include", "src/library.py")
        self.assertTrue((self.main / "src/library.py").exists())

    def test_preview_does_not_contact_origin(self):
        self.change_config()
        self.git(self.dev, "remote", "set-url", "origin", str(self.root / "missing-origin.git"))
        self.invoke("--dry-run")
        self.assertEqual(self.sha(self.main), self.before)

    def test_public_readme_cannot_reintroduce_the_university_link(self):
        self.write(self.dev, "README.md", README + f"\nAnother copy: [{UNI}](./{UNI})\n")
        self.commit(self.dev)
        self.invoke("--no-push", expected=1)
        self.assertEqual(self.sha(self.main), self.before)
        self.assertNotIn(UNI, (self.main / "README.md").read_text())

    def test_private_artifacts_cannot_be_included(self):
        for path in (UNI, "AGENTS.md", "docs/private.md"):
            with self.subTest(path=path):
                self.assertIn("Not an eligible public", self.invoke("--no-push", "--include", path, expected=1))
                self.assertEqual(self.sha(self.main), self.before)

    def test_private_additions_do_not_create_empty_publications(self):
        self.write(self.dev, "scripts/new_tool.py", "PRIVATE = True\n")
        self.write(self.dev, "results/EXP-001/smoke/result.json", "{}\n")
        self.commit(self.dev)
        self.invoke("--no-push")
        self.assertEqual(self.sha(self.main), self.before)
        self.assertFalse((self.main / "scripts").exists())

    def test_real_deletions_require_review(self):
        self.git(self.dev, "rm", "src/library.py")
        self.commit(self.dev)
        self.assertIn("--allow-deletions", self.invoke("--dry-run"))
        self.assertIn("--allow-deletions", self.invoke("--no-push", expected=1))
        self.assertEqual(self.sha(self.main), self.before)
        self.invoke("--no-push", "--allow-deletions")
        self.assertFalse((self.main / "src/library.py").exists())
        self.assertTrue((self.dev / UNI).exists())
        self.assert_clean()

    def test_rename_transfers_only_the_selected_new_path(self):
        self.git(self.dev, "mv", "src/library.py", "src/renamed.py")
        self.commit(self.dev)
        output = self.invoke("--no-push", "--include", "src/renamed.py", "--allow-deletions")
        self.assertIn("R100", output)
        self.assertTrue((self.main / "src/renamed.py").exists())
        self.assertFalse((self.main / "src/library.py").exists())

    def test_primary_pdf_cannot_be_removed(self):
        self.git(self.dev, "rm", PAPER)
        self.commit(self.dev)
        self.assertIn("primary PDF", self.invoke("--no-push", "--allow-deletions", expected=1))
        self.assertEqual(self.sha(self.main), self.before)

    def test_conflicting_readmes_stop_before_touching_main(self):
        self.write(self.dev, "README.md", README.replace("Old result summary.", "Dev result summary."))
        self.commit(self.dev)
        original = (self.main / "README.md").read_text().replace("Old result summary.", "Main result summary.")
        self.write(self.main, "README.md", original)
        before = self.commit(self.main, "public wording")
        self.assertIn("Conflicting public/dev edits", self.invoke("--no-push", expected=1))
        self.assertEqual(self.sha(self.main), before)
        self.assertEqual((self.main / "README.md").read_text(), original)
        self.assert_clean()

    def test_substantive_main_only_fix_must_be_backported(self):
        self.write(self.main, "config.py", "VALUE = 3\n")
        before = self.commit(self.main, "main fix")
        self.assertIn("Bring the substantive main-only changes", self.invoke("--no-push", expected=1))
        self.assertEqual(self.sha(self.main), before)
        self.write(self.dev, "config.py", "VALUE = 3\n")
        self.commit(self.dev)
        self.invoke("--no-push")
        self.assertEqual(self.sha(self.main), before, "backport alone must not create an empty publication")

    def test_invalid_python_and_json_fail_before_publication(self):
        for name, content in (("config.py", "this is not valid Python !!!\n"),
                              ("results/metrics.json", "{broken json}\n")):
            with self.subTest(path=name):
                old = (self.dev / name).read_bytes()
                self.write(self.dev, name, content)
                self.commit(self.dev)
                self.assertIn("Invalid public artifact", self.invoke("--no-push", expected=1))
                self.assertEqual(self.sha(self.main), self.before)
                self.write(self.dev, name, old)
                self.commit(self.dev)

    def test_symlink_cannot_publish_private_contents(self):
        (self.dev / "config.py").unlink()
        (self.dev / "config.py").symlink_to("docs/private.md")
        self.commit(self.dev)
        self.assertIn("non-regular file", self.invoke("--no-push", expected=1))
        self.assertEqual(self.sha(self.main), self.before)

    def test_commit_body_is_not_accepted(self):
        self.change_config()
        self.invoke("--message", "subject\n\nbody", expected=2)
        self.assertEqual(self.sha(self.main), self.before)
        self.invoke("--no-push", "--message", "update paper")
        self.assertEqual(self.git(self.main, "show", "-s", "--format=%B", "HEAD").strip(), "update paper")

    def test_local_publication_can_be_pushed_by_rerunning(self):
        source = self.change_config()
        self.invoke("--no-push")
        head = self.sha(self.main)
        self.assertEqual(self.sha(self.origin, "main"), self.before)
        self.invoke(cwd=self.main)
        self.assertEqual(self.sha(self.main), head)
        self.assertEqual(self.sha(self.origin, "main"), head)
        self.assertEqual(self.sha(self.origin, "dev"), source)
        self.assert_clean()

    def test_atomic_push_failure_is_retriable_without_a_new_commit(self):
        source = self.change_config()
        hook = self.origin / "hooks/pre-receive"
        hook.write_text("#!/bin/sh\ncat >/dev/null\nexit 1\n")
        hook.chmod(0o755)
        self.assertIn("Push not verified", self.invoke(expected=1))
        head = self.sha(self.main)
        self.assertNotEqual(head, self.before)
        self.assertEqual(self.sha(self.origin, "main"), self.before)
        self.assertEqual(self.sha(self.origin, "dev"), self.base)
        hook.unlink()
        self.invoke()
        self.assertEqual(self.sha(self.main), head)
        self.assertEqual(self.sha(self.origin, "main"), head)
        self.assertEqual(self.sha(self.origin, "dev"), source)
        self.assert_clean()

    def test_remote_main_ahead_is_not_overwritten(self):
        self.change_config()
        tree = self.git(self.main, "rev-parse", "HEAD^{tree}").strip()
        remote_change = self.git(self.main, "commit-tree", tree, "-p", self.before,
                                 data=b"remote update\n").strip()
        self.git(self.main, "push", "origin", f"{remote_change}:refs/heads/main")
        self.assertIn("ahead or diverged", self.invoke(expected=1))
        self.assertEqual(self.sha(self.main), self.before)
        self.assertEqual(self.sha(self.origin, "main"), remote_change)

    def test_unknown_remote_commit_requires_fetch_without_local_changes(self):
        self.change_config()
        tree = self.git(self.origin, "rev-parse", "main^{tree}").strip()
        remote_change = self.git(self.origin, "-c", "user.name=Remote Test",
                                 "-c", "user.email=remote@example.invalid", "commit-tree", tree,
                                 "-p", self.before, data=b"remote-only commit\n").strip()
        self.git(self.origin, "update-ref", "refs/heads/main", remote_change, self.before)
        self.assertIn("Run git fetch origin", self.invoke(expected=1))
        self.assertEqual(self.sha(self.main), self.before)
        self.assertEqual(self.sha(self.origin, "main"), remote_change)

    def test_unrelated_dev_history_is_rejected(self):
        tree = self.git(self.dev, "rev-parse", "HEAD^{tree}").strip()
        unrelated = self.git(self.dev, "commit-tree", tree, data=b"unrelated dev\n").strip()
        self.git(self.dev, "update-ref", "refs/heads/dev", unrelated, self.base)
        self.assertIn("last dev publication", self.invoke("--no-push", expected=1))
        self.assertEqual(self.sha(self.main), self.before)

    def test_another_publisher_holds_the_lock(self):
        common = Path(self.git(self.dev, "rev-parse", "--path-format=absolute", "--git-common-dir").strip())
        with (common / "publish-main.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertIn("already running", self.invoke("--dry-run", expected=1))
        self.assertEqual(self.sha(self.main), self.before)


if __name__ == "__main__":
    unittest.main()
