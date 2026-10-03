# Camp QMIND MVP: Golden Gate, Audited

This is the plan for our one-day demo. The full nine-month plan is in [PROPOSAL.md](PROPOSAL.md), and if the
two disagree about method, the proposal wins.

## What we're building

We'll redo Anthropic's Golden Gate Bridge demo (Templeton et al., 2024) on a small open model. Then we'll
check the features the way its critics say you should, using our four-criteria protocol from
[PROPOSAL.md § 4.3](PROPOSAL.md#43-feature-validation) plus tests taken from the published critiques below.
The problem we want to catch is a label that sounds right but isn't.

An MVP is the smallest version of the project that still counts as the project. For us that means three
things: find a feature with an SAE, test whether it does what its label says, and steer the model with it.
Training our own SAEs, reasoning models and circuit tracing are all out for today.

We picked this because it runs the same pipeline as the real project on a model that fits on a free Colab GPU,
and because a model that won't stop talking about a bridge makes a good demo. It also matters for the real
project. If we can't trust feature labels, none of our later claims about reasoning features hold up, so
auditing labels has to come first.

## Choices

- **No SAE training.** We load a pretrained one. Training is Phase 0 of the real project and takes far longer
  than a day.
- **Model:** Qwen3-1.7B-Base with the Qwen-Scope residual-stream SAEs (32K width), loaded from the
  [Qwen-Scope collection](https://huggingface.co/collections/Qwen/qwen-scope). It is small enough for our GPU
  (roughly 3.5 GB in bf16, verify), and subject to change. We looked at Qwen3-0.6B first, but Qwen-Scope
  [has no SAE for it](https://arxiv.org/html/2605.11887v1) and we aren't training one today. Gemma Scope only
  works with Gemma, so it is out. Verify that the SAEs load through SAELens as the model cards claim.
- **Features:** a bridge or landmark feature for the demo, and a Queen's University feature that we tune both
  up and down and audit. We search for them rather than assuming an index.
- **No reasoning models today.** That's the real project. The MVP is just the method.

Things marked (verify) come from memory or a single search and need checking against the live docs. In
particular, confirm that the Qwen-Scope SAEs load as described and that SAELens has moved to Decode Research.
Qwen-Scope may not be on Neuronpedia, so published labels may not exist for it.

## The criticism

Feature labels usually come from the top-activating examples, or from an LLM's summary of them. That can go
wrong. The kind of failure we have in mind is a feature labeled "water" whose top examples are mostly
drowning. We didn't find that exact case documented, so we don't build on it. Instead we test the failures
that are documented, listed next.

### What the public critiques say

From a web search in October 2026. We read the papers marked "read" at abstract or summary level only. For the
others we saw only the title and a search snippet, so open them before they go on a slide.

**Labels are shaky**

- Explanations tend to be too broad and ignore polysemanticity, and averaging interpretability over random
  latents can hide the problem, since it doesn't penalize frequent features that aren't very interpretable
  ([EleutherAI auto-interp write-up](https://www.lesswrong.com/posts/AhG3RJ6F5KvmKmAkd/open-source-automated-interpretability-for-sparse),
  snippet only).
- Top-activating examples are correlational evidence. A vision study found that for non-localized features,
  neither top images nor activation highlights gave a coherent reading
  ([Causal Interpretation of SAE Features in Vision](https://arxiv.org/html/2509.00749v1), snippet only).
  A label built from top examples can fit them and still be wrong about the feature.
- One explanation can describe many different features, so the label doesn't tell them apart
  ([Descriptive Collision in SAE Auto-Interpretability](https://arxiv.org/pdf/2605.12874), read).
- Existing explanations can be revised and falsified with held-out tests
  ([Revising and Falsifying SAE Feature Explanations](https://openreview.net/pdf?id=OJAW2mHVND), title only,
  could not open it). This is the closest prior work to our audit, so it is the first thing to read.

**Features are less clean than they look**

- Feature absorption: with hierarchical concepts, the SAE learns "A" plus "B except A", so a feature fires on
  95% of its cases and fails on an arbitrary 5% where a narrower feature absorbed them
  ([A is for Absorption](https://openreview.net/forum?id=Wzav8fesTL), Chanin et al.; snippet only). Bricken
  et al. (2023) already describe feature splitting.
- Wrong sparsity gives wrong features ([Sparse but Wrong](https://arxiv.org/pdf/2508.16560), title only). One
  more reason to use a published SAE today and not train a quick one.

**Steering and usefulness**

- Prompting beat every representation-level method on AxBench, and SAEs scored far below even simple
  difference-in-means (0.165 vs 0.239 average steering score in the snippet we saw). The authors blame part of
  it on concept lists that lean toward token-level concepts
  ([AxBench](https://arxiv.org/abs/2501.17148), snippet only).
- A later paper reports SAEs can get close to LoRA on AxBench when features are picked and labeled with a
  supervised pipeline, so this is still being argued
  ([Steering LLMs? Actually, SAEs can outperform simple baselines](https://arxiv.org/html/2605.31183v1),
  snippet only).
- Google DeepMind's interpretability team reported negative results for SAEs on downstream tasks and
  deprioritised SAE research
  ([post](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9),
  title only, the page returned 403 for us).

**Conceptual objections to Golden Gate specifically**

- Some commentary argues that "the Golden Gate Bridge feature" is a geometric artifact of optimization and
  not a discrete unit
  ([The Golden Gate Illusion](https://medium.com/@bulanramai2558/the-golden-gate-illusion-why-sparse-autoencoders-saes-misunderstand-the-physics-of-ai-8bf6cdc52928)).
  It is an opinion post, not a study. Cite it as a view, not as evidence.

**How this shapes the audit**

- A falsification test of the label (the falsification critique), a neighbor-collision test (the collision
  critique) and an absorption test (Chanin et al.), all described under "Critique audit" below.
- Per-subgroup specificity and sensitivity test the label directly.
- The prompting baseline in the steering comparison is there because of AxBench.
- The random-feature control and the sweep guard against reading too much into one clamped output.

We aren't trying to settle any of this today. We want to show that a cheap audit can catch a bad label on a
real feature, and that we can run the same audit on every feature we claim later.

## Workstreams

### 1. Data: probe sets

Small text sets, about 30 to 50 items each, built in a Colab notebook. Every item should be one a person has
read.

- For the bridge, a target set (sentences about the Golden Gate Bridge) and a near-miss set that is close in
  topic but shouldn't fire: other bridges, other SF landmarks, fog.
- For Queen's University, a target set (campus, Kingston, Queen's students and clubs, Queen's sports) and a
  near-miss set that shares the word or the place: Queen Elizabeth, the band Queen, queen bees, chess queens,
  other Ontario universities, other things in Kingston. Queen's is an ambiguous string, so this set matters.
- For the absorption test, a set of general university text (other schools, admissions, campus life without
  naming Queen's), so we can check the general feature on both general and Queen's-specific sentences.
- A held-out set for the falsification test. We write it after we have a label (see the audit), so it can't
  be tuned to the examples that produced the label.
- A neutral set of unrelated text, as the baseline.

We'll write a short note on how each set was made, either alongside the files or in the notebook.

### 2. Model: finding features

In a Colab GPU notebook:

1. Load the model and a pretrained residual-stream SAE at a middle layer (Qwen3-1.7B-Base and a Qwen-Scope
   SAE by default). Per [AGENTS.md](AGENTS.md), use SAELens and TransformerLens, and prefer the newer
   TransformerLens bridge API for new code (verify that it supports Qwen3).
2. Run the target sets through the model, encode with the SAE, and rank features by mean activation on the
   target set minus mean activation on the neutral set.
3. For the top candidates, record top tokens and examples, and read the published Neuronpedia label if there is
   one (verify).
4. Pick one bridge feature and one Queen's feature, plus a general "university" feature for the absorption
   test, and write down the layer and index.

Qwen-Scope only comes in 32K width for 1.7B, so the knob to turn is the layer. Qwen3-1.7B is the base model,
so steer with plain-text prompts ("Q: ... A:") rather than a chat template.

If there's no clean bridge feature at that layer, try other layers before changing anything else.
If that fails too, use some other clean single-entity feature and say that's what we did. A 1.7B model may
know little about Queen's specifically. If the best candidate is really "Canadian university" or "royalty",
that is a finding about the label, so record it and don't force the match.

We should end up with a table of candidates: index, layer, SAE width, top tokens, label.

### 3. Validation and steering

This takes the place of Tracking in the Camp QMIND outline.

**Four-criteria check** on each audit feature:

- *Specificity:* when it fires, is the label true? Measure precision against the labeled sets, per subgroup.
- *Sensitivity:* when the label is true, does it fire? Measure recall, per subgroup.
- *Causal effect:* clamp it high on neutral prompts and score what the model says (next section).
- *Not a single neuron:* check whether one or two residual dimensions explain the activation. We still need to
  decide how.

**Steering.** Clamp the bridge feature to a multiple of its max observed activation while generating. The
original used 10x on Claude 3 Sonnet. Our model will need its own multiple, since too high gives gibberish.
Run the same neutral prompts (say, "What is your favorite way to spend a weekend?") under four conditions:

1. No steering.
2. Feature clamped.
3. A system prompt telling the model to talk about the bridge. Critics say SAE steering has to beat this, so
   we report it.
4. A random feature with similar activation scale, clamped to the same multiple.

**Tuning Queen's up and down.** Same clamping hook, run in both directions on the Queen's feature:

- *Up:* clamp to a few multiples of its max activation and run neutral prompts ("What should I study?",
  "Describe a good place to spend four years"). We expect the model to start bringing up Queen's or Kingston.
- *Down:* clamp to zero while the prompt is about Queen's ("Tell me about Queen's University in Kingston").
  If the feature matters causally, the answer should get vaguer, drop Queen's-specific details, or drift
  toward a generic university. Compare with the same prompt unsteered and with a random feature zeroed.
- Check for spillover in both directions: does turning it up also push Queen Elizabeth, the band or Ontario
  in general, and does turning it down hurt answers about other universities? Spillover is the same kind
  of evidence as the audit below, since it shows what the feature actually encodes.
- Sweep the multiple (for example 0, 0.5x, 1x, 2x, 5x of max) and show the curve instead of one setting. A
  dose-response curve is better evidence than a single cherry-picked output.

**Critique audit.** Run on the Queen's feature, and on the bridge feature if there's time. Each test is our
own small version of a published critique, not a reproduction of the paper.

1. *Falsification* (in the spirit of the falsification paper, which we haven't been able to read yet). Get
   a label from the feature's top-activating examples, written by an LLM or taken from the Qwen-Scope
   materials if one exists. Ask what the label predicts should and shouldn't fire, write those sentences by
   hand as the held-out set, and run them. Note every miss and revise the label.
2. *Collision* (descriptive collision). Take the ten or so features most similar to ours (decoder cosine
   similarity, or co-activation on the probe sets), label them the same way, and check whether the labels
   actually separate them. Then check whether our feature behaves differently from its neighbors on the
   near-miss set.
3. *Absorption* (Chanin et al.). Queen's University sits under "university", so check whether the general
   university feature stays quiet on Queen's sentences where the Queen's feature fires, while firing on other
   schools. That pattern is what absorption looks like.
4. *Per-subgroup specificity and sensitivity* on the target, near-miss and general sets, as in the
   four-criteria check.
5. Write a label based on what we measured, put it next to the original, and report the gap. If the original
   holds up, say so. A clean audit is still a result.

For scoring, start with a keyword rate for the target concept and read every sample by hand, since there won't
be many. Add an LLM judge only if there's time, and spot-check it.

The output is one notebook and a short results note at `docs/mvp_results.md`. Per [AGENTS.md](AGENTS.md), no
number goes in the note unless a run produced it, and anything we didn't run gets marked as not run.

### 4. Presentation

Keep it short, since the demo does most of the work.

1. Team intro.
2. Why: models are opaque, and the tools for opening them are only as good as their labels.
3. How: SAE features, the four-criteria audit, steering.
4. Live demo: the bridge-obsessed model, a Queen's slider that goes up and down, then the critique audit,
   ending on the measured-label table. The slider is the part the audience can relate to, so make it
   interactive if we can (a notebook widget over the clamp multiple).
5. Next: reasoning features, which is the full proposal.

Record a backup of the demo in case Colab or the GPU fails on the day.

## Schedule

Adjust to how long the day actually is.

1. Environment: Colab GPU, model and SAE load, one prompt encoded. Done when sparse activations print.
2. Probe sets (Data) and feature search (Model), in parallel. Done when the candidate table exists.
3. Steering the bridge feature. Done when clamped output clearly differs from baseline.
4. Critique audit and four-criteria numbers. Done when the per-subgroup table and measured label exist.
5. Results note and slides. Done when we've rehearsed the demo once start to finish.

## Done means

- A bridge feature found, with layer and index.
- Clamped output compared with baseline, prompt baseline and random-feature control, on the same prompts.
- Queen's feature tuned up and down across a sweep, with spillover checked.
- Queen's feature put through the falsification, collision and absorption tests, with per-subgroup
  specificity and sensitivity from a real run.
- A measured label written next to the original one.
- Results note and slides finished, demo rehearsed, backup recording saved.

## AI usage

Use AI for boilerplate and for drafting probe sentences, but read what it gives you. A person checks every
probe item, every number in the results note comes from a run, and every citation gets opened and confirmed
before it goes on a slide.
