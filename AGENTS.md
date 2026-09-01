# AGENTS.md

Guidance for AI coding agents (Claude Code, Codex, etc.) and human contributors working in this repo.
It exists because the repo is currently pre-implementation: there's no code yet to infer conventions
from, so this is where they're written down ahead of time.

## Where the project stands

This is a QMIND research project in the proposal stage. [PROPOSAL.md](PROPOSAL.md) is the living
planning document (motivation, research questions, methodology, timeline); [README.md](README.md) is
the public-facing summary. Implementation (SAE training, feature validation, causal interventions,
evaluation) has not started. When in doubt about scope or method, `PROPOSAL.md` is the source of truth. If you implement something that changes what it describes, update `PROPOSAL.md` and `README.md` in
the same change rather than letting them drift.

## Repository layout

```
PROPOSAL.md   : source of truth for research questions, methodology, timeline
README.md     : public summary, kept in sync with PROPOSAL.md
papers/       : reference PDFs cited in PROPOSAL.md; treat as read-only source material, don't edit
LICENSE       : MIT
```

As implementation lands, follow the methodology sections of the proposal (§4.2–4.6) into a layout like:

```
src/          : installable package code (SAE training, feature validation, intervention tooling)
experiments/  : run configs / entry points per workstream
notebooks/    : exploratory notebooks (see notebook policy below)
configs/      : experiment configuration (e.g. Hydra/YAML), not hardcoded constants in scripts
```

Don't invent a different top-level layout without a reason. Extend this one as workstreams need it.

## Conventions once code lands

- **Python**, formatted and linted consistently (black/ruff or equivalent, pick one tool and apply it
  repo-wide rather than mixing styles across workstreams).
- Type hints and docstrings on anything that isn't a one-off script.
- **Config-driven experiments**: hyperparameters (dictionary size, layer, learning rate, seed) belong
  in a config file or CLI arg, not hardcoded, so a run is reproducible from its config alone.
- **Notebooks**: fine for exploration, but anything that produces a checked-in result should live as a
  script under `experiments/`. If a notebook is committed, clear its outputs first; don't commit
  execution state or embedded plots.

## Data and model handling

Never commit model weights, activation caches, SAE checkpoints, or other large artifacts. Extend the
existing (Python-oriented) `.gitignore` rather than working around it. Use Weights & Biases artifacts or
the HuggingFace Hub for anything that needs to be shared or versioned.

## The "test suite" for research claims

There's no traditional test suite yet, but there is a rubric every feature claim must pass: the
four-criteria protocol from [PROPOSAL.md § 4.3](PROPOSAL.md#43-feature-validation): high specificity,
high sensitivity, an appropriate causal effect under intervention, and non-correspondence to any single
neuron. Any code or writeup asserting "this is a real feature" should be traceable to results against
all four, not just a correlational plot. Once a validation harness exists, treat it the way you'd treat
a test suite: don't merge a "new feature" claim that doesn't run through it.

## Expectations for agents specifically

- Prefer extending SAELens / TransformerLens abstractions over reinventing dictionary-learning or
  activation-hooking infrastructure from scratch; both are mature and already chosen in the proposal.
- When implementing a method from the proposal, cite the relevant section/paper in a comment or commit
  message so the connection back to the methodology is traceable.
- Never fabricate experimental results, numbers, or plots. If a run hasn't been executed, say so, don't
  fill in a plausible-looking result.
- Keep `README.md` and `PROPOSAL.md` in sync with reality; don't let public-facing docs describe a
  method or timeline that's since changed.

For git workflow, PR process, and experiment-logging conventions, see [CONTRIBUTING.md](CONTRIBUTING.md).
