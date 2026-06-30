# Fine-Tuning Local LLMs to Write Git Commit Messages

Can a small, locally-run language model write better commit messages than a generic, prompted one? This project says yes — and shows that **task-specific fine-tuning beats raw model size** for a narrow task.

The goal: generate clean, [Conventional Commits](https://www.conventionalcommits.org/)–style messages from a `git diff`, entirely on-device. No code leaves the machine, inference is fast, and it works offline — so it can drop into a `prepare-commit-msg` git hook via [Ollama](https://ollama.com/).

📝 **Full write-up:** https://eliotbas.com/projects/commits-fine-tuning/

---

## What's in this repo

This repository holds the **data pipeline** — the part that mines and curates the training data. Fine-tuning itself runs in a Google Colab notebook (A100 + [Unsloth](https://github.com/unslothai/unsloth)) and isn't checked in here.

```
scrap-commit-messages.py   # mines (diff → commit message) pairs from local git clones
datasets/                  # versioned training sets (commits_v1/v2/v3.jsonl)
repos/                     # cloned source repos (gitignored)
```

### `scrap-commit-messages.py`

Extracts `(diff, commit message)` pairs from 13 large open-source TypeScript repos that follow Conventional Commits (Angular, Vite, Vue core, Nuxt, Prisma, tRPC, TypeORM, Vitest, and more). Curation is the hard part, and the script is opinionated about quality:

- **Drops noise** — bot/automation commits (Dependabot, Renovate, semantic-release…), merges, reverts, version bumps, and WIP/typo subjects.
- **Strips generated files** from each diff (lockfiles, `dist/`, `*.min.js`, `*.d.ts`, snapshots, generated protobufs/GraphQL…), then scopes the diff to the files a human actually wrote.
- **Enforces format** — every subject must match the Conventional Commits pattern (`type(scope)?!: subject`) and fall within sane length/word bounds.
- **Bounds diff size** — line, char, and changed-file caps so examples fit a small model's context.
- **Deduplicates** identical diffs across repos and caps how often any single subject can repeat.
- **Balances sources** — per-repo caps plus a seeded shuffle keep one repo from dominating the global subject cap.

It also prints dataset health checks: per-repo counts, commit-type distribution, message/diff length percentiles, file-extension coverage, and unique-subject ratio.

## The result

Across multiple small open-weight models (Qwen2.5-Coder 0.5B–3B, Qwen2.5, Llama-3.2-1B), LoRA fine-tuning (r=16, ~1% of params, ~20–30 min/model) improved **every structural metric** — ROUGE-L up 26–164%, Conventional Commits compliance near 100%. Most strikingly, a fine-tuned **0.5B** model outscored the un-tuned **3B** baseline.

Evaluation used a three-layer framework: ROUGE-L overlap, structural/format checks, and an LLM-as-judge (GPT-4o-mini) over a 200-example test set.

The full methodology, metrics, and limitations are in the [blog post](https://eliotbas.com/projects/commits-fine-tuning/).

## Reproducing the dataset

```bash
python scrap-commit-messages.py
```

This clones (or updates) each source repo under `repos/` and writes `datasets/commits_v3.jsonl`, with each line a chat-formatted training example:

```json
{
  "messages": [
    { "role": "user", "content": "<diff>" },
    { "role": "assistant", "content": "fix(router): ..." }
  ],
  "meta": { "repo": "angular", "hash": "..." }
}
```

Requires Python 3.10+ and `git`. No third-party dependencies.
