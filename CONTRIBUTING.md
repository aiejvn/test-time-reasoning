# Contributing

This is the team workflow doc for the QMIND `test-time-reasoning` project. For repo layout and coding
conventions (once implementation starts), see [AGENTS.md](AGENTS.md); this doc doesn't repeat those,
it covers how the team works together around them.

## Workstreams

Work is organized into the four workstreams from [PROPOSAL.md § 7](PROPOSAL.md#7-team-roles-and-training):

- **SAE training and infrastructure**
- **Feature validation**
- **Causal interventions**
- **Evaluation and diagnostics**

Pick up work through whichever workstream it belongs to, and loop in that workstream's owner before
starting something that spans more than one. Assignment to a workstream isn't fixed; people move
between them as the project's needs shift over the year, and someone can end up contributing to more
than one.

## Git workflow

- **Branches**: `workstream/short-description`, e.g. `sae-infra/activation-cache`,
  `feature-validation/specificity-metric`.
- **Commits**: describe what changed and why in the message body when it's not obvious from the diff
  alone: future-you (or a teammate picking up the workstream) is the audience.
- **PRs**: reviewed by the relevant workstream owner, with the project manager looped in on anything
  that touches methodology, scope, or the timeline in [PROPOSAL.md § 6](PROPOSAL.md#6-timeline-and-milestones).
- **Issues**: track work against the timeline's phases (Phase 0 infra, Phase 1 discovery/validation,
  Phase 2 interventions, Phase 3 cross-model generalization) so it's visible which phase is behind or
  ahead.

## Experiment logging

Every run that produces a result someone might cite (in a standup, a draft, or the paper) goes to
Weights & Biases, tagged with at minimum: model, layer, dictionary size, and seed. Untracked one-off
runs are fine for debugging, but don't build a claim on top of one; rerun it tracked first.

## Reproducibility

Pin dependencies (don't let two people's environments silently drift), and document the exact command
used to produce a result you're reporting. If a result can't be reproduced from the config + command
alone, treat that as a bug in the experiment, not an acceptable gap.

Every experiment's original launch command must be recorded somewhere, no fixed format required: a
W&B run note, a line in the run's log output, the PR/commit message that reports the result, a comment
in the config file, whatever fits the workflow. The only requirement is that it exists somewhere
findable, not that it lives in a particular place.

## Adding a reference paper

Drop the PDF in `papers/`, then add the citation to `PROPOSAL.md`'s References section and, if it
changes the framing of the project, to `README.md` as well. Don't let `papers/` accumulate PDFs that
aren't cited anywhere.

## Changing PROPOSAL.md

The proposal is a living document pre-funding, not a frozen spec, but it's also the thing an advisor
and funders will read, so changes to research questions, methodology, or timeline should come with a
short rationale (in the PR description is fine) and visibility to the faculty advisor before merging.
Small clarifications and typo fixes don't need this; substantive changes to scope or claims do.

## Communication

Weekly sync cadence and the team's primary async channel are set by the project manager. Check with
them if you're new. Keep discussion respectful and assume good faith; this is a research team working
on genuinely open problems, and being wrong in a discussion is normal, not a failure.
