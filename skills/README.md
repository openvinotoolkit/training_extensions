# Geti Agent Skills

Canonical, repo-specific agent skills for Geti. Skills are grouped by the part
of the repo they target so agents load the right paths and commands.

## Buckets

| Bucket          | Path                           | Scope                                                                                                                                                   |
| --------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Getitune**    | [`getitune/`](getitune/)       | `getitune`: models, recipes, training, export, optimization, inference, and the `getitune` CLI under `libraries/getitune/`.                             |
| **Application** | [`application/`](application/) | Application stack: `application/backend/` (FastAPI `geti`), `application/ui/` (React), the OpenAPI contract, the REST pipeline, and cross-cutting docs. |

Each bucket has its own skill list and `EVALUATION.md` scenarios.

## Layout

```text
skills/
├── README.md                 # this file — global authoring rules
├── getitune/
│   ├── README.md
│   ├── EVALUATION.md
│   └── <skill-name>/
│       └── SKILL.md
└── application/
    ├── README.md
    ├── EVALUATION.md
    └── <skill-name>/
        └── SKILL.md
```

Client adapters are **committed symlinks** so a fresh clone works for agents
(no setup step):

- `.claude/skills/<name>` → `../../skills/<bucket>/<name>`
- `.agents/skills/<name>` → `../../skills/<bucket>/<name>`

When you add or rename a skill, run
`python3 .github/scripts/skills/agent_skills.py sync` and commit the updated
symlinks. Pre-commit runs `sync` then `validate` when `skills/` changes. CI only
runs `validate` on what is in the PR (it does not regenerate symlinks). GitHub
may show `\ No newline at end of file` on symlink diffs; that is normal and
harmless.

**Windows:** enable
[Developer Mode](https://learn.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development)
or clone with `git config core.symlinks true` so Git checks out the symlinks. If
symlink creation fails, `agent_skills.py sync` falls back to a directory
junction for local use.

## Authoring standard

These skills follow the open [Agent Skills](https://agentskills.io) format. Write
to the **portable core** so a skill works across every agent, not just one.

### Portability rules

- **Frontmatter = the portable subset only:** `name` and `description` (and
  optionally `license`). These are understood everywhere.
- **Do not rely on vendor-only fields for behavior.** Default to model-invoked
  skills.
- Use forward-slash paths; no Windows backslashes.

### `name`

- Lowercase, hyphenated, matches the directory name, regex
  `^[a-z0-9]+(-[a-z0-9]+)*$`.
- Use the established Geti prefixes: `getitune-` for library (training library)
  skills, `geti-` for application and cross-cutting skills.
- Prefer gerund/verb phrasing (`getitune-training-a-model`,
  `geti-openapi-sync`).

### `description` (highest-leverage field)

This is what an agent matches against to decide whether to load the skill.

- **Third person**, always: "Trains and validates…", never "I can…".
- State **what it does and when to use it**, with concrete triggers (CLI names,
  class names, file paths, task families).
- **One trigger per distinct branch** — collapse synonyms. Keep it under 1024
  characters.

### Body

- **Be concise; assume the model is smart.** Keep `SKILL.md` well under 500
  lines.
- **Ground every claim in real paths** for that bucket
  (`libraries/getitune/src/getitune/...`, `application/backend/app/...`,
  `application/ui/src/...`) and real commands. No invented flags.
- **Numbered workflow steps, each ending in a checkable completion criterion**
  ("Done when: …") so the agent can tell done from not-done.
- **Match freedom to fragility:** high freedom (prose) for open tasks; low
  freedom ("run exactly this, don't add flags") for fragile/destructive ops like
  export, quantization, or migrations.
- **Feedback loops** for quality-critical work: run → validate → fix → repeat.
- **Progressive disclosure:** push long detail into `references/*.md`, linked one
  level deep from `SKILL.md`.

### Before you ship

- Description has both _what_ and _when_, third person, distinct triggers.
- Every path/command verified against the current tree.
- Steps have checkable completion criteria.
- Runs offline by default; steps needing downloads are marked.
- Passes at least three scenarios in the bucket's `EVALUATION.md`.

## Add a new skill

```bash
BUCKET=getitune  # or application
NAME=getitune-my-workflow
mkdir -p "skills/$BUCKET/$NAME"
$EDITOR "skills/$BUCKET/$NAME/SKILL.md"
python3 .github/scripts/skills/agent_skills.py sync
```

Then dry-run the workflow end-to-end and fix any step where an agent could stall
or guess.
