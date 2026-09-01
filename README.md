# Test-Time Reasoning

### Mechanistic Interpretability of Test-Time Reasoning in Large Language Models

A QMIND research project (2026-2027) building a sparse-autoencoder account of uncertainty, premise
commitment, and error recovery in reasoning models, asking what's happening inside a model in the
moment it goes confidently, fluently wrong.

The full research proposal lives in [PROPOSAL.md](PROPOSAL.md); this README is a working summary.

## The problem

Reasoning models (the DeepSeek-R1 line, the OpenAI o-series, Qwen's math-tuned models) solve hard
problems by "thinking" in long chains of intermediate steps before answering. We can watch those chains
from the outside: read the tokens, measure output entropy, score step correctness. But we have no
account of the internal computation producing them. That gap matters because these models fail in a
distinctive and dangerous way, not hesitantly, but confidently. A fluent, internally coherent chain
asserts a false intermediate result and rationalizes everything built on top of it. Token-level entropy,
the standard surface signal for "the model is unsure," is blind to this by construction: a
confidently-wrong model has a sharp, not flat, output distribution.

Two research lines bear on this and have so far developed in isolation: a **behavioral** line that
manages reasoning from the outside (process-reward models, entropy-gated branching, fine-tuned
self-refinement), and a **mechanistic interpretability** line that decomposes model internals into
interpretable features and circuits, but only for static content, never for a reasoning trajectory in
motion. This project connects them: train sparse autoencoders (SAEs) on a reasoning model's activations
while it reasons, and use the behavioral line's signals as instrumentation for finding, validating, and
causally testing the internal features that actually govern branching, commitment, and recovery.

## Research questions

- **RQ1, feature discovery.** What interpretable features are active during branching, premise
  commitment, backtracking, and error recovery, and how is each causally validated rather than merely
  correlated?
- **RQ2, miscalibrated confidence.** What is the mechanistic signature of confident-but-wrong
  reasoning, the low-entropy failures surface signals can't detect? Does an internal
  confidence/commitment representation predict these failures better than output-distribution statistics?
- **RQ3, invariance.** How invariant are these features across model families, scale, and training
  regime (reinforcement-learned vs. distilled/supervised)?
- **RQ4, intervention.** Do targeted interventions on identified features (amplifying a
  self-correction feature, suppressing a premise-commitment feature) steer reasoning toward
  self-correction and better accuracy, predictably?

See [PROPOSAL.md § 3](PROPOSAL.md#3-research-questions) for the full framing.

## Approach

1. **Train SAEs** on the residual-stream activations of open, reasoning-specialized models: a
   Qwen math/reasoning model and a DeepSeek-R1-distilled model, chosen so a reinforcement-learned vs.
   distilled comparison (RQ3) is possible within one study. (The exact checkpoints are a moving target
   as new releases ship; [PROPOSAL.md § 4.1](PROPOSAL.md#41-models) is the source of truth for whatever
   is currently targeted.)
2. **Validate every feature** against a four-criteria protocol: high specificity, high sensitivity, an
   appropriate causal effect under intervention, and non-correspondence to any single neuron, so
   claims are defensible rather than anecdotal.
3. **Locate the moments that matter** using two external signals as instrumentation: token-level
   entropy/varentropy to flag uncertainty and branching, and single-pass step-correctness annotation to
   label correct vs. incorrect steps.
4. **Causally intervene**, using ablation, amplification/clamping, and cross-prompt activation patching,
   to test whether steering a candidate feature changes reasoning outcomes in the predicted direction.
5. **Fall back to circuits** (attribution graphs, cross-layer transcoders) where a single feature can't
   explain a behavior.
6. **Evaluate** on GSM8K and ARC-AGI, plus purpose-built diagnostics isolating branching, commitment,
   and recovery.

Full methodology: [PROPOSAL.md § 4](PROPOSAL.md#4-methodology).

## Tech stack

SAELens and TransformerLens for SAE training and model internals, HuggingFace for model/dataset
access, Weights & Biases for experiment tracking, and the open Circuit Tracer library for the
circuit-level fallback. This is the current plan, not a fixed commitment: expect the specific tools and
even model choices to shift over the course of the year as workstreams hit real needs the initial plan
didn't anticipate.

## Repository structure

Right now, this repo holds the proposal and reference material:

```
PROPOSAL.md   - the full research proposal (motivation, methodology, timeline, team)
papers/       - reference PDFs for the works cited in the proposal
LICENSE       - MIT
```

As implementation starts, code will follow the layout and conventions described in
[AGENTS.md](AGENTS.md) (SAE training/infra, feature validation, causal interventions, and
evaluation/diagnostics, matching the workstreams below).

## Timeline

Nine months, July 2026 to March 2027, staffed lighter over the summer and at full strength once the
academic term starts.

| Phase | Months | Focus |
|---|---|---|
| 0 | Jul-Aug | Infra: SAE training pipeline, four-criteria validation harness, entropy/step-correctness instrumentation |
| 1 | Sep-Nov | Feature discovery (RQ1); begin confidence study (RQ2); methodology workshop paper |
| 2 | Dec-Jan | Causal interventions and confident-but-wrong analysis (RQ2, RQ4) |
| 3 | Feb-Mar | Cross-model generalization (RQ3); consolidate strongest result for a top-venue submission |

Full detail: [PROPOSAL.md § 6](PROPOSAL.md#6-timeline-and-milestones).

## Team

A QMIND team led by a project manager (research direction, milestones, writing), supported by a
research supervisor, with the rest of the team organized into four workstreams: **SAE training and
infrastructure**, **feature validation**, **causal interventions**, and **evaluation and diagnostics**.
See [PROPOSAL.md § 7](PROPOSAL.md#7-team-roles-and-training) for supervision and training details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the team's git workflow, experiment-logging conventions, and
how to propose changes to the proposal itself. See [AGENTS.md](AGENTS.md) for repo conventions aimed at
both human contributors and AI coding agents.

## License

[MIT](LICENSE).
