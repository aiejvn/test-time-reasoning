# Test-Time Reasoning

### Mechanistic Interpretability of Test-Time Reasoning in Large Language Models

A QMIND research project (2026-2027) on what happens inside a reasoning model when it goes confidently,
fluently wrong. The full research proposal is in [PROPOSAL.md](PROPOSAL.md); the project overview is
[further down](#project-overview).

## This branch: DeepSeek-R1-Distill-Llama-8B infrastructure

The main deliverable of this branch is infrastructure for running **DeepSeek-R1-Distill-Llama-8B** (32 layers,
4096-dimensional residual stream) in-process on one GPU, with four capabilities:

1. **Run the model** on GSM8K (math) and HumanEval+ (code), with batched greedy generation, automatic grading, and
   seeded dev/test splits.
2. **Read activations:** capture any layer's output (resid_post) at every token, or pooled per problem (last prompt
   token, response mean, last response token), at several layers in one pass.
3. **Apply changes during generation:** hook any layer and edit its output on every forward pass (add a direction,
   project a direction out, or any function you write). Baseline and edited runs share one code path, so their
   settings match exactly.
4. **Linear probes:** train per-layer classifiers on those activations, then use their directions as intervention
   targets. This is in [`linear_probe/`](linear_probe/).

Everything is one Python module, [`src/reasoning_env.py`](src/reasoning_env.py). There's no server. The model takes
about 16 GB in bf16, so it fits on a 23 GB A10G with room for generation.

### Where things are

| path | what's there |
|---|---|
| [`src/reasoning_env.py`](src/reasoning_env.py) | **Core infrastructure.** Model loading, activation capture, intervention hooks, benchmark loading, prompting and grading. |
| [`src/sae_env.py`](src/sae_env.py) | Model-agnostic helpers reused by the above (forward hooks, batched chat generation, logging); also the Gemma helpers. |
| [`experiments/baseline.py`](experiments/baseline.py), [`configs/baseline.yaml`](configs/baseline.yaml) | Baseline run: generate, grade, save every response and the activations at layers 4/19/29/30. |
| [`linear_probe/`](linear_probe/) | Linear probes and probe-guided interventions. Code, configs, a [README](linear_probe/README.md) and the [results report](linear_probe/REPORT.md). |
| [`experiments/sae_explore.py`](experiments/sae_explore.py), [`experiments/sae_report.py`](experiments/sae_report.py), [`configs/sae_explore.yaml`](configs/sae_explore.yaml) | SAE feature exploration with pretrained SAEs for this model. Results in [`reports/2026-11-20-sae-exploration.md`](reports/2026-11-20-sae-exploration.md). |
| [`coding_eval/`](coding_eval/) | Jacob's Dockerized EvalPlus evaluator for saved HumanEval+ outputs (merged from `jacobpowerr/qmind-coding-eval`). Not yet wired to our outputs. |
| `artifacts/` (gitignored) | Every run's outputs: responses, grades, activations, probe directions, intervention runs. |
| [`notebooks/`](notebooks/) | Earlier Gemma-3-4B-IT SAE failure-mode notebooks (separate track, [below](#gemma-sae-failure-mode-experiments)). |

### Quick start

```python
import sys; sys.path.insert(0, "src")
import reasoning_env as R

model, tok = R.load_model()                                   # DeepSeek-R1-Distill-Llama-8B, bf16, cuda
items = R.load_benchmark("gsm8k", n_dev=200, n_test=300)["dev"][:8]

# Generate + grade (rows: id, correct, pred, n_resp_tokens, truncated, ...; gens: prompt_ids, resp_ids, text)
rows, gens = R.run_benchmark(model, tok, "gsm8k", items)

# Read activations: {pool: {layer: [n_problems, 4096]}}
acts = R.extract_activations(model, tok, gens, layers=[4, 19, 29, 30], pools=["prompt_last", "resp_mean"])

# Apply changes during generation, at any layer. Same call, same settings, plus `edits`.
u = acts["prompt_last"][29].mean(0)                           # any [4096] direction
rows, gens = R.run_benchmark(model, tok, "gsm8k", items, edits={29: R.steer(u, alpha=3.9)})     # h + 3.9·unit(u)
rows, gens = R.run_benchmark(model, tok, "gsm8k", items, edits={29: R.ablate(u)})               # h − (h·û)û
rows, gens = R.run_benchmark(model, tok, "gsm8k", items,
                             edits={29: R.steer(R.random_direction(4096, seed=1), alpha=3.9)})  # random control

# Lower level: hook any forward/generate call yourself
with R.intervene(model, {19: lambda h: h * 1.1}, positions="generated"):   # edit only generated tokens
    out = model.generate(...)
with R.capture_resid_post(model, [19, 29]) as buf:                         # buf[layer] = list of [B, T, 4096]
    model.model(**inputs)
```

- `alpha` is in activation-length units, and typical per-token lengths grow with depth: about 4 at layer 4, 16 at
  layer 19, and 38 to 43 at layers 29 and 30.
- `positions="all"` (the default) edits the prompt and every generated token. `"generated"` edits only generated
  tokens.

### Baseline run

```bash
source .venv/bin/activate
python experiments/baseline.py --config configs/baseline.yaml               # full run
python experiments/baseline.py -o run_name=smoke -o 'splits=[dev]' \
  -o benchmarks.gsm8k.n_dev=8 -o benchmarks.humanevalplus.n_dev=8            # ~7 min smoke test
```

**Settings:**
- **Splits:** seeded and disjoint. GSM8K is 200 dev / 300 test, from its 1319-problem test set. HumanEval+ is
  64 dev / 99 test; HumanEval/32 is excluded because its HF test fails the reference solution.
- **Decoding:** greedy, with DeepSeek's recommended prompt ("Please reason step by step, and put your final answer
  within \boxed{}"), up to 4096 new tokens.
- **Prompt ablation:** `prompt_variant=plain` drops the step-by-step instruction.
- **Overrides:** any config key can be overridden with `-o dotted.key=value`. `benchmarks.<b>.limit=N` runs only the
  first N problems of each split.

**Grading:**
- **GSM8K:** the last `\boxed{}` after `</think>` (or the last number) must match exactly.
- **HumanEval+:** the last python block after `</think>` runs against the base and plus tests (30 s timeout).
- Responses truncated before they finish count as wrong.

**Outputs** go to `artifacts/baseline/<run_name>/`:
- `<bench>_<split>.jsonl`: every problem's prompt, response, grade, length and truncation flag.
- `.meta.json`: config, launch command and summary.
- `_acts.pt`: the activations.
- `summary.json`.

Runs are resumable per split. Set `wandb.enabled=true` to log to W&B.

### Results so far

- **Linear probes and interventions:** [`linear_probe/REPORT.md`](linear_probe/REPORT.md).
- **SAE feature exploration:** [`reports/2026-11-20-sae-exploration.md`](reports/2026-11-20-sae-exploration.md).

## Project overview

### The problem

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

### Research questions

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

### Approach

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

### Tech stack

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
src/reasoning_env.py - DeepSeek-R1-Distill-Llama-8B: multi-layer activation extraction, intervention hooks, GSM8K/HumanEval+ eval
experiments/       - baseline.py (DeepSeek baseline), sae_explore.py + sae_report.py (SAE exploration)
configs/           - baseline.yaml, sae_explore.yaml
linear_probe/      - linear probes + probe-guided interventions (code, configs, README, REPORT)
coding_eval/       - Dockerized EvalPlus HumanEval+ evaluator (merged from jacobpowerr/qmind-coding-eval)
reports/           - SAE exploration report
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
