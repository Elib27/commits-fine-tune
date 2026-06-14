"""Extract (diff, commit message) pairs from local git clones for LLM fine-tuning.

Designed for repos that follow Conventional Commits (angular, vite, vue, botpress).
Filters out bot commits, lockfile-only diffs, oversized diffs, and low-quality
subject lines. Caps per-repo contribution so no single repo dominates.
"""

import json
import re
import subprocess
import hashlib
import random
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent
REPOS_DIR = ROOT / "repos"
OUTPUT_FILE = ROOT / "datasets" / "commits_v2.jsonl"

# Source repos to mine, keyed by their local directory name under repos/.
REPOS = {
    "angular": "https://github.com/angular/angular.git",
    "botpress": "https://github.com/botpress/botpress.git",
    "commitlint": "https://github.com/conventional-changelog/commitlint.git",
    "cypress": "https://github.com/cypress-io/cypress.git",
    "nuxt": "https://github.com/nuxt/nuxt.git",
    "prisma": "https://github.com/prisma/prisma.git",
    "tanstack-query": "https://github.com/TanStack/query.git",
    "trpc": "https://github.com/trpc/trpc.git",
    "twenty": "https://github.com/twentyhq/twenty.git",
    "typeorm": "https://github.com/typeorm/typeorm.git",
    "typescript-eslint": "https://github.com/typescript-eslint/typescript-eslint.git",
    "vite": "https://github.com/vitejs/vite.git",
    "vitest": "https://github.com/vitest-dev/vitest.git",
    "vuejs-core": "https://github.com/vuejs/core.git",
}

PER_REPO_TARGET = 1000
CLONE_DEPTH = 10000
MIN_DIFF_LINES = 6
MAX_DIFF_LINES = 300
MAX_CHANGED_FILES = 6
MAX_DIFF_CHARS = 12_000  # guards against few-but-very-long-line diffs
MIN_SUBJECT_LEN = 15
MAX_SUBJECT_LEN = 150
MAX_SUBJECT_OCCURRENCES = 5
RANDOM_SEED = 42  # fixed seed so the shuffled subject-cap selection is reproducible

# Strip trailing PR/issue refs that aren't knowable from a diff.
# Examples: " (#1234)", " (GH-12)", " [#9876]"
TRAILING_REF_RE = re.compile(r"\s*[\(\[](?:#|GH-)\d+[\)\]]\s*$")

# Require Conventional Commits format: "type(scope)?!: subject".
CONVENTIONAL_RE = re.compile(
    r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore)"
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
        r"codegen",
        r"auto-?merge",
        r"nx-bot",
        r"allcontributors",
    ]
]

BAD_SUBJECT_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^bump\b",
        r"^chore\(release\)",
        r"^merge\b",
        r"^revert\b",
        r"^release\b",
        r"^v?\d+\.\d+\.\d+",  # version-only
        r"^update (the )?(readme|changelog|deps|dependencies|version)",
        r"^typo\b",
        r"^wip\b",
        r"^fix\s*:\s*$",
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
        r"\.d\.ts$",
        r"(^|/)coverage/",
        r"__generated__",
        r"\.graphql\.ts$",
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


def diff_fingerprint(diff: str) -> str:
    # Strip line numbers and file paths, keep only +/- lines
    lines = [
        l
        for l in diff.splitlines()
        if l.startswith(("+", "-")) and not l.startswith(("+++", "---"))
    ]
    normalized = "\n".join(lines).lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()


def gather_candidates(repo: Path, target: int):
    """Return filtered candidate examples for a repo (no cross-repo dedup yet).

    Cross-repo concerns (identical-commit dedup and the global subject cap) are
    applied later in main() against the shuffled pool, so the capped slots for a
    shared subject aren't all claimed by whichever repo is processed first.
    """
    candidates = []
    for commit_hash, author, subject in list_commits(repo, CLONE_DEPTH):
        if len(candidates) >= target:
            break
        if is_bot_author(author):
            continue
        if is_bad_subject(subject):
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

        candidates.append(
            {
                "subject_key": subject.lower().strip(),
                "fingerprint": diff_fingerprint(diff),
                "example": {
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
                },
            }
        )
    return candidates


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


def type_distribution_check(dataset):
    type_re = re.compile(r"^(\w+)[\(:]")
    types = []
    repos = []
    for ex in dataset:
        subject = ex["messages"][1]["content"]
        repo = ex["meta"]["repo"]
        m = type_re.match(subject)
        if m:
            types.append(m.group(1).lower())
        repos.append(repo)
    print()
    print("DISTRIBUTION")
    print("type")
    print(Counter(types).most_common())
    print("Repo")
    print(Counter(repos).most_common())


def sample_random_examples(dataset):
    for ex in random.sample(dataset, 20):
        print("DIFF:", ex["messages"][0]["content"][:300])
        print("MSG: ", ex["messages"][1]["content"])
        print("---")


def dedup_check(dataset):
    subjects = [ex["messages"][1]["content"].lower().strip() for ex in dataset]
    print(f"Total: {len(subjects)}")
    print(f"Unique subjects: {len(set(subjects))}")


def scope_coverage(dataset):
    extensions = Counter()
    for ex in dataset:
        diff = ex["messages"][0]["content"]
        for match in re.finditer(r"diff --git a/\S+\.(\w+)", diff):
            extensions[match.group(1)] += 1
    print(extensions.most_common(20))


def print_data_shape(dataset, per_repo_counts):
    print()
    print(f"Total: {len(dataset)} examples -> {OUTPUT_FILE}")
    for name, count in per_repo_counts.items():
        print(f"  {name}: {count}")
    msg_lens = [len(ex["messages"][1]["content"]) for ex in dataset]
    diff_lens = [len(ex["messages"][0]["content"]) for ex in dataset]
    print_length_summary("Commit message", msg_lens)
    print_length_summary("Diff prompt", diff_lens)

    type_distribution_check(dataset)
    # sample_random_examples(dataset)
    dedup_check(dataset)
    scope_coverage(dataset)


def main():
    REPOS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in REPOS.items():
        ensure_repo(name, url)
    print()

    candidates = []
    for repo in sorted(p for p in REPOS_DIR.iterdir() if p.is_dir()):
        print(f"=> {repo.name}")
        repo_candidates = gather_candidates(repo, PER_REPO_TARGET)
        print(f"   gathered {len(repo_candidates)} candidates")
        candidates.extend(repo_candidates)

    # Shuffle the pooled candidates so the global subject cap is filled fairly
    # across repos instead of being claimed by whichever repo is processed first.
    random.seed(RANDOM_SEED)
    random.shuffle(candidates)

    all_examples = []
    per_repo_counts: Counter = Counter()
    global_seen: set[tuple[str, str]] = set()
    subject_global_counts: Counter = Counter()
    for cand in candidates:
        key = (cand["subject_key"], cand["fingerprint"])
        if key in global_seen:
            continue
        if subject_global_counts[cand["subject_key"]] >= MAX_SUBJECT_OCCURRENCES:
            continue
        global_seen.add(key)
        subject_global_counts[cand["subject_key"]] += 1
        all_examples.append(cand["example"])
        per_repo_counts[cand["example"]["meta"]["repo"]] += 1

    with OUTPUT_FILE.open("w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    # Reload the existing dataset so we can inspect it without re-mining.
    # all_examples = []
    # per_repo_counts = {}
    # with OUTPUT_FILE.open() as f:
    #     for line in f:
    #         line = line.strip()
    #         if not line:
    #             continue
    #         ex = json.loads(line)
    #         all_examples.append(ex)
    #         repo = ex["meta"]["repo"]
    #         per_repo_counts[repo] = per_repo_counts.get(repo, 0) + 1

    print_data_shape(all_examples, per_repo_counts)


if __name__ == "__main__":
    main()
