# Test-Time Reasoning

### Mechanistic Interpretability of Test-Time Reasoning in Large Language Models

A QMIND research project (2026-2027) on what happens inside a reasoning model when it goes confidently,
fluently wrong. The full research proposal is in [PROPOSAL.md](PROPOSAL.md); the project overview is
[further down](#project-overview).

## This branch: reading and editing DeepSeek-R1's internals

This branch builds the tooling to look inside **DeepSeek-R1-Distill-Llama-8B** while it solves problems, and to
change what's inside while it runs. It then uses that tooling for a first experiment:

> Can we tell, from the model's internal state, whether it's about to answer correctly? If so, can we nudge
> that internal state to make it answer correctly more often, on math and on code?

This section explains **what** was done and **how**, assuming you know how transformers and LLMs work but haven't
seen this project. Detailed results are in [`linear_probe/REPORT.md`](linear_probe/REPORT.md) and
[`reports/2026-11-20-sae-exploration.md`](reports/2026-11-20-sae-exploration.md).

### Background: the four ideas this work uses

**1. The residual stream.** A transformer keeps one vector per token, here 4096 numbers, and each of its 32 layers
reads that vector and adds to it. The vector after layer L, called *resid_post* at layer L, is the model's working
state for that token at that depth. Everything here is measured on, or applied to, these vectors. We look at
layers 4 (early), 19 (middle; the layer used in the "I Have Covered All the Bases Here" paper), and 29 and 30
(late, but not the last layer).

**2. Hooks.** PyTorch can run a function every time a layer produces its output. A hook can *read* that output
(to record activations) or *replace* it (to intervene). Because it runs on every forward pass, it also runs on
every generated token during `generate()`. That lets us change the model's internals while it writes, with no
retraining and no change to its weights.

**3. Linear probes.** A probe is a plain logistic-regression classifier trained to predict a label (here, "did the
model answer this problem correctly?") from a layer's activation vector. If the probe works on problems it wasn't
trained on, the information is linearly readable at that layer. The probe's weight vector is a direction in the
4096-dimensional space, roughly "the direction that separates correct from wrong". A probe that works shows the
information is *present*, not that the model *uses* it. Using it is what interventions test.

**4. Steering and ablation.** Given a direction u, there are two simple edits to the layer output h at each token:
- **Steering:** h ← h + α·u, pushing the state along u by an amount α.
- **Ablation:** h ← h − (h·u)u, deleting the part of h that lies along u.

If steering toward "correct" raises accuracy more than steering along a **random** direction of the same size, the
direction has a causal effect beyond just disturbing the model. That comparison against random is the key control.

**Also used: sparse autoencoders (SAEs).** An SAE is a separately trained network that rewrites a 4096-number
activation as a sparse combination of tens of thousands of "features". Usually only a few dozen are active per
token, and each tends to mean something recognizable (a word, a topic, a reasoning move). We use pretrained SAEs for
this model to ask which features differ between correct and wrong answers.

### What was done, step by step

```
 ┌─────────────────────────┐   ┌───────────────────────┐   ┌──────────────────────────┐
 │ 1. Infrastructure       │──▶│ 2. Baseline           │──▶│ 3. Linear probes         │
 │ load model, hooks to    │   │ run the unedited      │   │ predict "correct?" from  │
 │ read/edit any layer,    │   │ model on math + code, │   │ each layer's activations │
 │ graders                 │   │ save answers + acts   │   │ → direction u            │
 └─────────────────────────┘   └───────────────────────┘   └────────────┬─────────────┘
                                          │                            ▼
                                          │              ┌──────────────────────────┐
                                          │              │ 4. Interventions         │
                                          │              │ steer / ablate along u   │
                                          │              │ vs. random-direction     │
                                          │              │ control, on unseen probs │
                                          ▼              └──────────────────────────┘
                               ┌───────────────────────┐
                               │ 5. SAE exploration    │
                               │ which SAE features    │
                               │ differ correct vs.    │
                               │ wrong?                │
                               └───────────────────────┘
```

#### Step 1. Infrastructure ([`src/reasoning_env.py`](src/reasoning_env.py))

**What:** one Python module that runs the model on one GPU and gives you four operations:
- generate and grade answers;
- record any layer's activations;
- edit any layer's output during generation;
- load the two benchmarks.

**How:**
- **Model:** loaded with Hugging Face `transformers` in bf16. It takes about 16 GB, so it fits on a 23 GB A10G with
  room for generating long reasoning traces.
- **Reading activations:** a forward hook on decoder layer L captures its output (resid_post).
  `extract_activations` re-runs a saved prompt and response through the model and summarizes each problem in one of
  three ways ("pools"):
  - `prompt_last`: the vector at the last prompt token, right after `<think>` and before any reasoning.
  - `resp_mean`: the mean over the response tokens.
  - `resp_last`: the last response token.
- **Editing:** `intervene(model, {layer: fn})` installs a hook that replaces the layer's output with `fn(output)` on
  every forward pass. `steer(u, α)` and `ablate(u)` are the two edits above.
- **One code path:** `run_benchmark(..., edits=...)` is the same function with or without edits. Baseline and edited
  runs therefore use identical prompts, batching and decoding.
- **Checks:** a steering edit of 8·u changed the layer output by exactly 8·u. Ablation removed 99.9% of the component
  along u. An edit of size 0 reproduced the unedited output token-for-token.

**Benchmarks and grading:**
- **GSM8K (math word problems):** the model's last `\boxed{}` answer must equal the reference number exactly.
- **HumanEval+ (write a Python function):** the extracted code is run against the original and the stricter "plus"
  unit tests in a subprocess.
- **Prompting:** both use DeepSeek's recommended "Please reason step by step" prompt with greedy decoding, up to 4096
  new tokens.
- **Splits:** problems are split, with a fixed seed, into **dev** (used to build and tune things) and **test** (held
  out for the final check).

#### Step 2. Baseline ([`experiments/baseline.py`](experiments/baseline.py))

**What:** run the unedited model and keep everything: the prompt, the full response including its reasoning, whether
it was correct, its length, and activations at layers 4, 19, 29 and 30.

**How:** `python experiments/baseline.py --config configs/baseline.yaml`. Outputs go to `artifacts/baseline/`
(gitignored).

**Result:**

| | problems | correct | hit 4096-token limit |
|---|---|---|---|
| GSM8K dev | 200 | 83.0% | 9.5% |
| HumanEval+ dev | 32 | 62.5% | 37.5% |

*Terms:*
- **dev:** problems used for building and tuning, as opposed to held-out test problems.
- **problems:** how many were run. For HumanEval+, this is the first 32 of 64 dev problems.
- **correct:** share graded right.
- **hit 4096-token limit:** share of responses cut off before finishing, which are graded on what was written so
  far.

Many wrong answers are the model still reasoning when it runs out of tokens: 17 of 34 wrong on GSM8K and 11 of 12
on HumanEval+.

#### Step 3. Linear probes ([`linear_probe/probe.py`](linear_probe/probe.py))

**What:** for each layer, train a probe to predict "will this answer be correct?" from the activations saved in
step 2.

**How:**
- Logistic regression with standardized inputs, balanced classes and strong regularization (4096 inputs, ~200
  examples).
- **Score:** 5-fold cross-validated AUROC, so every problem is scored by a probe that didn't train on it. AUROC is
  the chance that a random correct problem scores above a random wrong one: 0.5 is chance, 1.0 is perfect.

**Result:**
- **At the last prompt token** (before the model has written anything), AUROC is 0.65–0.73 on math and 0.65–0.77 on
  code. Both peak at layer 29, so some of the outcome is visible before the model reasons.
- **Over the whole response**, AUROC reaches 0.85–0.91, but that summary can see length and truncation.
- The GSM8K probe direction at layer 29 becomes the direction u for step 4.

#### Step 4. Interventions ([`linear_probe/intervene.py`](linear_probe/intervene.py))

**What:** re-generate answers with the layer-29 output edited at every token, and count which problems the edit
fixed and which it broke. Four conditions run on the same problems in the same batches:

| condition | edit | role |
|---|---|---|
| no edit | none | baseline |
| probe steering | h + α·u | the actual test |
| random control | h + α·r, r random | is any push of this size enough? |
| probe ablation | h − (h·u)u | does the model need this direction? |

*Terms:*
- **h:** the layer-29 output vector at one token.
- **u:** the probe direction, unit length, pointing toward "correct".
- **r:** a random unit direction, with fixed seed 1.
- **α:** push size.
- **h·u:** how far h extends along u (dot product), so (h·u)u is that component.

**How:**
- α is set relative to the layer's scale: 0.1 × 38.65, the median length of one token's vector at layer 29.
- The strength was chosen on 16 dev problems, then the final run used problems never touched before: 20 GSM8K and
  12 HumanEval+ test problems.
- Fixed and broken counts are compared with an exact McNemar test.

**Result:**
- Probe steering made **no net difference**: GSM8K 90% → 90%, HumanEval+ 83.3% → 83.3%.
- The random control did as well or better.
- Ablation: GSM8K 95% (1 fixed, 0 broken), HumanEval+ 66.7% (0 fixed, 2 broken).
- Nothing is statistically significant at this sample size.
- **Noise:** with greedy decoding, merely changing which problems share a batch flipped about 3 of 16 outcomes in a
  dev check.

Full tables, every problem's outcome under every condition, and caveats are in
[`linear_probe/REPORT.md`](linear_probe/REPORT.md).

#### Step 5. SAE exploration ([`experiments/sae_explore.py`](experiments/sae_explore.py))

**What:** using pretrained SAEs for this exact model, list the features whose activation differs most between
correct and wrong answers. For each feature, show where it fires most strongly, to tell a "reasoning" feature from
one that just fires on a word like "Wait".

**How:**
- The step-2 prompts and responses are re-run through the model and every token is encoded with the SAE.
- Features are ranked by Cohen's d, the difference in group means divided by the pooled standard deviation, both at
  the last prompt token and averaged over the response.
- Each feature links to its Neuronpedia dashboard where one exists.
- SAEs used:
  - **Llama Scope R1-Distill** at layers 4, 19, 29 and 30 (32,768 features each).
  - **The SAE from Galichin et al. 2025** at layer 19 (65,536 features).
- Reconstruction quality on our data, measured as the cosine between the SAE's reconstruction and the real
  activation: 0.82–0.90 for Llama Scope and 0.84–0.85 for the paper's SAE.

**Result:** per-SAE feature tables are in [`reports/2026-11-20-sae-exploration.md`](reports/2026-11-20-sae-exploration.md).
These are correlations on dev data. None of these features has been intervened on or validated.

#### Also on this branch: [`coding_eval/`](coding_eval/)

Jacob's evaluator, merged from `jacobpowerr/qmind-coding-eval`. It grades saved HumanEval+ outputs with the official
EvalPlus harness inside a locked-down Docker container and compares baseline against intervention runs. It isn't
connected to our outputs yet. It uses its own splits (20 dev / 40 held-out) and its own input format, so an exporter
is needed.

### Not done yet

- **Larger held-out runs:** more problems, several strengths, several random seeds, other layers.
- **Negative steering (pushing away from "correct").**
- **Input ablation:** the prompt without "reason step by step". This is implemented as `prompt_variant=plain` but
  hasn't been run.
- **SAE-guided interventions:** steering along individual SAE features.
- **Grading through `coding_eval/`.**

### Where things are

| path | what's there |
|---|---|
| [`src/reasoning_env.py`](src/reasoning_env.py) | Step 1: model loading, activation capture, intervention hooks, benchmarks, grading |
| [`src/sae_env.py`](src/sae_env.py) | Model-agnostic helpers used by the above (hooks, batched generation, logging), plus the earlier Gemma helpers |
| [`experiments/baseline.py`](experiments/baseline.py), [`configs/baseline.yaml`](configs/baseline.yaml) | Step 2: baseline run |
| [`linear_probe/`](linear_probe/) | Steps 3–4: [README](linear_probe/README.md), `probe.py`, `intervene.py`, configs, [REPORT.md](linear_probe/REPORT.md) |
| [`experiments/sae_explore.py`](experiments/sae_explore.py), [`experiments/sae_report.py`](experiments/sae_report.py), [`configs/sae_explore.yaml`](configs/sae_explore.yaml) | Step 5: SAE exploration and its report builder |
| [`reports/2026-11-20-sae-exploration.md`](reports/2026-11-20-sae-exploration.md) | Step 5 results |
| [`coding_eval/`](coding_eval/) | Jacob's Dockerized EvalPlus evaluator |
| `artifacts/` (gitignored) | Every run's raw outputs: responses, grades, activations, probe directions, intervention runs |
| [`notebooks/`](notebooks/) | Earlier, separate Gemma-3-4B-IT SAE notebooks ([below](#gemma-sae-failure-mode-experiments)) |

*Terms:*
- **acts:** activations, the saved layer-output vectors.
- **gitignored:** kept locally, never committed.

### Using the infrastructure

```python
import sys; sys.path.insert(0, "src")
import reasoning_env as R

model, tok = R.load_model()                                   # DeepSeek-R1-Distill-Llama-8B, bf16, cuda
items = R.load_benchmark("gsm8k", n_dev=200, n_test=300)["dev"][:8]

# Generate and grade. rows: id, correct, pred, n_resp_tokens, truncated, ...; gens: prompt_ids, resp_ids, text
rows, gens = R.run_benchmark(model, tok, "gsm8k", items)

# Read activations: {pool: {layer: [n_problems, 4096]}}
acts = R.extract_activations(model, tok, gens, layers=[4, 19, 29, 30], pools=["prompt_last", "resp_mean"])

# Edit a layer during generation: the same call plus `edits`
u = acts["prompt_last"][29].mean(0)                           # any [4096] direction
rows, gens = R.run_benchmark(model, tok, "gsm8k", items, edits={29: R.steer(u, alpha=3.9)})     # h + 3.9·unit(u)
rows, gens = R.run_benchmark(model, tok, "gsm8k", items, edits={29: R.ablate(u)})               # remove u component
rows, gens = R.run_benchmark(model, tok, "gsm8k", items,
                             edits={29: R.steer(R.random_direction(4096, seed=1), alpha=3.9)})  # random control

# Lower level: your own hook around any forward or generate call
with R.intervene(model, {19: lambda h: h * 1.1}, positions="generated"):   # edit only generated tokens
    out = model.generate(...)
with R.capture_resid_post(model, [19, 29]) as buf:                         # buf[layer] = list of [B, T, 4096]
    model.model(**inputs)
```

- **Scale of α:** typical per-token vector lengths grow with depth: about 4 at layer 4, 16 at layer 19, and 38–43 at
  layers 29 and 30. Choose α relative to the layer.
- **Positions:** `positions="all"` (the default) edits the prompt and every generated token. `"generated"` edits only
  generated tokens.

### Running the experiments

```bash
source .venv/bin/activate
python experiments/baseline.py --config configs/baseline.yaml                         # step 2
python linear_probe/probe.py --config linear_probe/configs/probe.yaml                 # step 3
python linear_probe/intervene.py --config linear_probe/configs/intervene_sweep.yaml   # step 4, pick strength on dev
python linear_probe/intervene.py --config linear_probe/configs/intervene.yaml         # step 4, held-out test
python experiments/sae_explore.py --config configs/sae_explore.yaml -o sae=llamascope_l19   # step 5, one SAE per run
python experiments/sae_report.py                                                      # step 5 report
```

- **Overrides:** any config value can be overridden with `-o dotted.key=value`, for example
  `-o benchmarks.humanevalplus.limit=32` to run only the first 32 problems of each split.
- **Resuming:** runs pick up where they left off.
- **Launch commands:** each run's exact command is saved next to its outputs in `artifacts/`.
- **Smoke test:** `python experiments/baseline.py -o run_name=smoke -o 'splits=[dev]' -o benchmarks.gsm8k.n_dev=8 -o
  benchmarks.humanevalplus.n_dev=8`, about 7 minutes.

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
