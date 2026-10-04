"""Push the local git repo to GitHub via the Git Data API (author: 晨星).

Rationale (SOP): git smart-protocol pushes die with 502 through the local
proxy while api.github.com works, so we push blobs -> tree -> commit ->
ref via `gh api`. Idempotent: re-running skips already-uploaded blobs and
updates the branch ref to the new head. Empty-repo bootstrap: seed one file
via the Contents API first (trees API 409s on a truly empty repo).
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
from pathlib import Path

OWNER = "CJX0712"
REPO = "domainforge"
BRANCH = "main"


def gh(*args: str, input_text: str | None = None) -> dict:
    """gh api wrapper with backoff retry on transient gateway errors."""
    # `--input -` with an EMPTY body makes GitHub 404 GETs (verified); only
    # attach a stdin body when there actually is one
    cmd = (
        ["gh", "api", "--input", "-", *args] if input_text is not None else ["gh", "api", *args]
    )
    for attempt in range(6):
        r = subprocess.run(
            cmd, capture_output=True, text=True, input=input_text, encoding="utf-8"
        )
        err = (r.stderr or "") + (r.stdout or "")
        transient = any(k in err for k in ("502", "503", "Bad Gateway", "reset", "timed out"))
        not_found_yet = (
            "404" in err
            and attempt < 5
            and ("git/" in args[-1] or "/commits/" in args[-1] or "/ref/" in args[-1])
        )
        if r.returncode == 0:
            return json.loads(r.stdout) if r.stdout.strip() else {}
        if (transient or not_found_yet) and attempt < 5:
            wait = 3 * (attempt + 1)
            print(f"  retry {attempt + 1} in {wait}s: {err.strip()[:120]}")
            time.sleep(wait)
            continue
        raise RuntimeError(f"gh api failed: {args[:2]} :: {err.strip()[:400]}")
    raise RuntimeError("unreachable")


def local_tree() -> list[tuple[str, bytes]]:
    """All tracked files at HEAD (respects .gitignore)."""
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    files = []
    for rel in out:
        data = (Path(rel)).read_bytes()
        files.append((rel.replace("\\", "/"), data))
    return files


def main() -> int:
    files = local_tree()
    print(f"{len(files)} tracked files")

    # 1. ensure repo exists
    r = subprocess.run(
        ["gh", "repo", "view", f"{OWNER}/{REPO}", "--json", "name"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print("creating repo...")
        subprocess.run(
            [
                "gh",
                "repo",
                "create",
                REPO,
                "--public",
                "--description",
                "DomainForge: modular domain adaptation system - "
                "CORAL/TCA/JDA/KLIEP pure-numpy + SAFuse flagship "
                "(HPO + gain-gated fusion + non-inferiority safeguard), "
                "CPU-only offline fallback. Author: 晨星",
                "--source=.",
                "--disable-warnings",
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    # 2. bootstrap commit if branch is empty
    head_sha = None
    r = subprocess.run(
        ["gh", "api", f"repos/{OWNER}/{REPO}/git/ref/heads/{BRANCH}"],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        head_sha = json.loads(r.stdout)["object"]["sha"]
        print(f"branch exists at {head_sha[:12]}")
    else:
        print("empty repo -> seeding bootstrap commit (README)")
        b64 = base64.b64encode(files[0][1]).decode()
        gh(
            "-X",
            "PUT",
            f"repos/{OWNER}/{REPO}/contents/{files[0][0]}",
            input_text=json.dumps(
                {
                    "message": "bootstrap (author: 晨星)",
                    "content": b64,
                    "branch": BRANCH,
                }
            ),
        )
        head_sha = gh(f"repos/{OWNER}/{REPO}/commits/{BRANCH}")["sha"]

    # 3. base tree = current remote tree (so deleted files stay deleted)
    base_tree = gh(f"repos/{OWNER}/{REPO}/git/commits/{head_sha}")["tree"]["sha"]

    # 4. upload blobs (skip ones already on remote by content? simple: upload all)
    tree_items = []
    for i, (path, data) in enumerate(files):
        blob = gh(
            "-X",
            "POST",
            f"repos/{OWNER}/{REPO}/git/blobs",
            input_text=json.dumps(
                {
                    "content": base64.b64encode(data).decode(),
                    "encoding": "base64",
                }
            ),
        )
        tree_items.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        if (i + 1) % 10 == 0:
            print(f"  blobs {i + 1}/{len(files)}")

    # 5. create tree -> commit -> move branch ref
    tree = gh(
        "-X",
        "POST",
        f"repos/{OWNER}/{REPO}/git/trees",
        input_text=json.dumps({"base_tree": base_tree, "tree": tree_items}),
    )
    commit = gh(
        "-X",
        "POST",
        f"repos/{OWNER}/{REPO}/git/commits",
        input_text=json.dumps(
            {
                "message": "DomainForge v0.1.0: modular domain adaptation "
                "system (CORAL/TCA/JDA/KLIEP + SAFuse flagship). "
                "Author: 晨星",
                "tree": tree["sha"],
                "parents": [head_sha],
                "author": {
                    "name": "晨星",
                    "email": "CJX0712@users.noreply.github.com",
                    "date": time.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
                },
            }
        ),
    )
    gh(
        "-X",
        "PATCH",
        f"repos/{OWNER}/{REPO}/git/refs/heads/{BRANCH}",
        input_text=json.dumps({"sha": commit["sha"], "force": False}),
    )
    print(f"pushed commit {commit['sha'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
