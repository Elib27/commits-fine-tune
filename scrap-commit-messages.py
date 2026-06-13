"""Extract (diff, commit message) pairs from local git clones for LLM fine-tuning.

Designed for repos that follow Conventional Commits (angular, vite, vue, botpress).
Filters out bot commits, lockfile-only diffs, oversized diffs, and low-quality
subject lines. Caps per-repo contribution so no single repo dominates.
"""

import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent
REPOS_DIR = ROOT / "repos"
OUTPUT_FILE = ROOT / "commits.jsonl"

# Source repos to mine, keyed by their local directory name under repos/.
REPOS = {
    "angular": "https://github.com/angular/angular.git",
    "botpress": "https://github.com/botpress/botpress.git",
    "vite": "https://github.com/vitejs/vite.git",
    "vuejs-core": "https://github.com/vuejs/core.git",
    "nuxt": "https://github.com/nuxt/nuxt.git",
    "typescript-eslint": "https://github.com/typescript-eslint/typescript-eslint.git",
    "typeorm": "https://github.com/typeorm/typeorm.git",
    "supabase": "https://github.com/supabase/supabase.git",
    "cypress": "https://github.com/cypress-io/cypress.git",
}

PER_REPO_TARGET = 2000
CLONE_DEPTH = 15000
MIN_DIFF_LINES = 6
MAX_DIFF_LINES = 300
MAX_CHANGED_FILES = 6
MAX_DIFF_CHARS = 15_000  # guards against few-but-very-long-line diffs
MIN_SUBJECT_LEN = 15
MAX_SUBJECT_LEN = 150

# Strip trailing PR/issue refs that aren't knowable from a diff.
# Examples: " (#1234)", " (GH-12)", " [#9876]"
TRAILING_REF_RE = re.compile(r"\s*[\(\[](?:#|GH-)\d+[\)\]]\s*$")

# Require Conventional Commits format: "type(scope)?!: subject".
CONVENTIONAL_RE = re.compile(
    r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(\([^)]+\))?!?: ",
    re.IGNORECASE,
)

BOT_AUTHOR_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"dependabot",
        r"renovate",
        r"greenkeeper",
        r"\[bot\]",
        r"github-actions",
        r"snyk-bot",
        r"semantic-release",
    ]
]

BAD_SUBJECT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^bump\b",
        r"^chore\(deps\)",
        r"^merge\b",
        r"^revert\b",
        r"^release\b",
        r"^v?\d+\.\d+\.\d+",  # version-only
        r"^update (the )?(readme|changelog|deps|dependencies|version)",
        r"^typo\b",
        r"^wip\b",
        r"^fix(\s|\:)*(typo|lint|format)?\s*$",
        r"^\.+$",
        r"^\s*$",
    ]
]

GENERATED_FILE_PATTERNS = [
    re.compile(p)
    for p in [
        r"package-lock\.json$",
        r"yarn\.lock$",
        r"pnpm-lock\.yaml$",
        r"Gemfile\.lock$",
        r"poetry\.lock$",
        r"composer\.lock$",
        r"go\.sum$",
        r"Cargo\.lock$",
        r"\.min\.(js|css)$",
        r"\.map$",
        r"(^|/)CHANGELOG(\.md)?$",
        r"(^|/)vendor/",
        r"(^|/)dist/",
        r"(^|/)build/",
        r"(^|/)node_modules/",
        r"\.pb\.go$",
        r"_pb2\.py$",
        r"\.snap$",
    ]
]


def is_bot_author(author: str) -> bool:
    return any(p.search(author) for p in BOT_AUTHOR_PATTERNS)


def is_bad_subject(subject: str) -> bool:
    s = subject.strip()
    if not CONVENTIONAL_RE.match(s):
        return True
    if not (MIN_SUBJECT_LEN <= len(s) <= MAX_SUBJECT_LEN):
        return True
    if s.isupper():
        return True
    if not re.search(r"[a-z]", s):
        return True
    if len(s.split()) < 3:
        return True
    if any(p.search(s) for p in BAD_SUBJECT_PATTERNS):
        return True
    return False


def is_generated_path(path: str) -> bool:
    return any(p.search(path) for p in GENERATED_FILE_PATTERNS)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        errors="replace",
    ).stdout


def run_git(*args: str, cwd: Path | None = None) -> None:
    """Run git for its side effects, streaming output and raising on failure."""
    subprocess.run(["git", *args], cwd=cwd, check=True)


def ensure_repo(name: str, url: str) -> None:
    """Clone the repo if missing, otherwise fetch and hard-reset to latest."""
    dest = REPOS_DIR / name
    if not dest.exists():
        print(f"=> cloning {name}")
        run_git("clone", "--depth", str(CLONE_DEPTH), url, str(dest))
    else:
        print(f"=> updating {name}")
        run_git("fetch", "--depth", str(CLONE_DEPTH), "origin", cwd=dest)
        branch = git(dest, "rev-parse", "--abbrev-ref", "HEAD").strip()
        run_git("reset", "--hard", f"origin/{branch}", cwd=dest)


def list_commits(repo: Path, n: int):
    """Yield (hash, author, subject) for the last n non-merge commits."""
    raw = git(
        repo,
        "log",
        "--no-merges",
        f"--max-count={n}",
        "--format=%H%x09%an%x09%s",
    )
    for line in raw.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            yield parts[0], parts[1], parts[2]


def changed_files(repo: Path, commit_hash: str):
    out = git(repo, "show", "--name-only", "--format=", commit_hash)
    return [f for f in out.splitlines() if f]


def scoped_diff(repo: Path, commit_hash: str, files: list[str]) -> str:
    # Limit the diff to non-generated files via pathspec.
    return git(repo, "show", "--stat", "-p", "--format=", commit_hash, "--", *files)


def extract_from_repo(repo: Path, target: int):
    examples = []
    seen_subjects = set()
    for commit_hash, author, subject in list_commits(repo, CLONE_DEPTH):
        if len(examples) >= target:
            break
        if is_bot_author(author):
            continue
        if is_bad_subject(subject):
            continue
        # de-dup near-identical subjects within a repo (catches release-bot style)
        key = subject.lower().strip()
        if key in seen_subjects:
            continue

        files = changed_files(repo, commit_hash)
        real_files = [f for f in files if not is_generated_path(f)]
        if not real_files:
            continue
        if len(real_files) > MAX_CHANGED_FILES:
            continue

        diff = scoped_diff(repo, commit_hash, real_files)
        line_count = diff.count("\n")
        if line_count < MIN_DIFF_LINES or line_count > MAX_DIFF_LINES:
            continue
        if len(diff) > MAX_DIFF_CHARS:
            continue

        clean_subject = TRAILING_REF_RE.sub("", subject).strip()
        if is_bad_subject(clean_subject):
            continue

        seen_subjects.add(key)
        examples.append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Generate a commit message for this diff:\n\n"
                            f"```diff\n{diff}\n```"
                        ),
                    },
                    {"role": "assistant", "content": clean_subject},
                ],
                "meta": {
                    "repo": repo.name,
                    "hash": commit_hash,
                },
            }
        )
    return examples


def percentiles(values: list[int], ps=(50, 75, 90, 95, 99)) -> dict[int, int]:
    """Nearest-rank percentiles for a list of ints."""
    s = sorted(values)
    last = len(s) - 1
    return {p: s[round(p / 100 * last)] for p in ps}


def print_length_summary(label: str, values: list[int]) -> None:
    if not values:
        return
    pct = percentiles(values)
    print(f"\n{label} char length (n={len(values)}):")
    print(f"  min {min(values)}  mean {sum(values) // len(values)}  max {max(values)}")
    print("  " + "  ".join(f"p{p}={v}" for p, v in pct.items()))


def main():
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in REPOS.items():
        ensure_repo(name, url)
    print()

    all_examples = []
    per_repo_counts = {}
    for repo in sorted(p for p in REPOS_DIR.iterdir() if p.is_dir()):
        print(f"=> {repo.name}")
        examples = extract_from_repo(repo, PER_REPO_TARGET)
        per_repo_counts[repo.name] = len(examples)
        print(f"   kept {len(examples)} examples")
        all_examples.extend(examples)

    with OUTPUT_FILE.open("w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print()
    print(f"Total: {len(all_examples)} examples -> {OUTPUT_FILE}")
    for name, count in per_repo_counts.items():
        print(f"  {name}: {count}")

    msg_lens = [len(ex["messages"][1]["content"]) for ex in all_examples]
    diff_lens = [len(ex["messages"][0]["content"]) for ex in all_examples]
    print_length_summary("Commit message", msg_lens)
    print_length_summary("Diff prompt", diff_lens)


if __name__ == "__main__":
    main()
