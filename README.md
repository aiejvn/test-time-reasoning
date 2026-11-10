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

```
PROPOSAL.md        - the full research proposal (motivation, methodology, timeline, team)
papers/            - reference PDFs for the works cited in the proposal
notebooks/         - Gemma SAE failure-mode experiments, one notebook per failure case (see below)
src/sae_env.py     - shared helpers: load Gemma, hook the residual stream, load Gemma Scope SAEs, generate, pool features
requirements.txt   - pinned dependencies
.env.example       - template for the local, gitignored .env (HF_TOKEN)
LICENSE            - MIT
```

Further code will follow the layout and conventions described in [AGENTS.md](AGENTS.md) (SAE
training/infra, feature validation, causal interventions, and evaluation/diagnostics, matching the
workstreams below).

## Gemma SAE failure-mode experiments

A side track of the SAE workstream (not part of [PROPOSAL.md](PROPOSAL.md)'s reasoning-trajectory work):
for failures of Gemma-3-4B-IT, do SAE features track the failure? Each notebook tries to elicit one
failure on as many examples as possible with greedy decoding, auto-labels it, and caches residual-stream
SAE features (Gemma Scope 2, layer 17, 16k, medium L0 by default) for analysis. Custom SAE training and the
full metrics (AUROC vs a dense probe, FDR, autointerp, ablation) are not built yet.

| notebook | failure | data |
|---|---|---|
| `notebooks/bad_code.ipynb` | incorrect code (unit tests); missed vulnerabilities | HumanEval, MBPP, DiverseVul |
| `notebooks/bias.ipynb` | stereotype-consistent answers; discrepant decisions | BBQ, Discrim-Eval |
| `notebooks/deception.ipynb` | lying when instructed or pressured | Geometry of Truth, MASK |

### Setup

Needs Python 3.12 and an NVIDIA GPU (developed on one A10G, 23 GB; Gemma takes about 8.6 GB in bf16).
`E.load_gemma` refuses to fall back to CPU.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then put your Hugging Face token in .env
```

### Models and data

- **Gemma-3-4B-IT** is gated. Accept the license on [its Hugging Face page](https://huggingface.co/google/gemma-3-4b-it)
  (manual approval), create a read token at https://huggingface.co/settings/tokens, and set `HF_TOKEN` in `.env`.
  `src/sae_env.py` loads `.env` on import, so notebooks need nothing else. The weights (about 8.6 GB) download to
  the Hugging Face cache on first use; to fetch them ahead of time:
  `set -a; source .env; set +a; hf download google/gemma-3-4b-it`.
- **Gemma Scope 2 SAEs** are ungated and download through SAELens on first load (`SAE_RELEASE = "gemma-scope-2-4b-it-res"`,
  ids like `layer_17_width_16k_l0_medium`). Available residual SAEs: layers 9, 17, 22, 29; widths 16k, 65k, 262k, 1m;
  L0 small, medium, big. The SAE for layer L must be fed layer L's resid_post, which `E.get_resid_post` does.
- **Datasets** download automatically on first run through `datasets`, except Geometry of Truth, whose CSVs are
  fetched from GitHub into `artifacts/data/`.

### Running

```bash
jupyter lab        # open a notebook in notebooks/ and run all cells
```

`QMIND_N` (default 200) sets examples per condition; a full run of all three notebooks at N=200 took about 30 minutes.
Every example's prompt, response and label is written to `artifacts/logs/<section>.jsonl` (plus `.meta.json`), and pooled
activations to `artifacts/*.pt`. `artifacts/` is gitignored; never commit it. Notebooks are committed without outputs:
run `nbstripout --install` once per clone.

### Known limitations of the current labelers

- DiverseVul triage: Gemma is at chance (about 52% accuracy), so failures mostly reflect the prompt, and the "functions labeled vulnerable" are noisy.
- Geometry of Truth: under the deceive prompt Gemma mostly gives a constant answer instead of inverting the truth, so the failure rate overstates real lying (only `sp_en_trans` inverts).
- MASK: the judge is Gemma itself and is noisy.
- Discrim-Eval: few failures (14 at N=200), and the baseline row differs in age as well as race.

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
