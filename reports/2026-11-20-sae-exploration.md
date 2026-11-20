# SAE feature exploration: DeepSeek-R1-Distill-Llama-8B

**Status:** 0 of 5 SAEs finished; still queued: Llama Scope, layer 4, Llama Scope, layer 19, Llama Scope, layer 29, Llama Scope, layer 30, Galichin et al. paper SAE, layer 19. This file is regenerated after each SAE finishes.

## What this is

A sparse autoencoder (SAE) rewrites a layer's 4096-number activation vector as a sparse combination of a much larger
set of "features" (32,768 or 65,536 here). Usually only a few dozen are active at any token, and each is meant to
correspond to one recognizable concept. This exploration asks: **which SAE features differ between problems the model
answers correctly and problems it gets wrong?** It covers layers 4, 19, 29 and 30, on math (GSM8K) and code (HumanEval+).

This is exploration, not a validated feature claim. A feature that differs between correct and wrong answers is a
correlate. Whether it causes anything is untested until we intervene on it. Under the project's four-criteria protocol
(specificity, sensitivity, causal effect, not a single neuron), none of these features has been validated.

## Data

The model's own outputs from the baseline run, unchanged model, greedy decoding, up to 4096 new tokens:

- **GSM8K dev:** 200 grade-school math word problems; 166 correct, 34 wrong.
- **HumanEval+ dev:** the first 32 of our 64 dev coding problems; 20 correct, 12 wrong.

These are the same problems the linear probes were trained on. No held-out problems are used here.

## Method

1. Each problem's prompt and the model's full response (reasoning included) are fed back through the model, and the
   chosen layer's output is recorded at every token.
2. The SAE encodes every token (except the first, a special token with an outlier activation).
3. Each problem gets two summaries:
   - **last prompt token:** the feature activations at the final prompt token, right after `<think>` and before the
     model writes anything. This reflects the model's state before reasoning.
   - **response mean:** each feature's activation averaged over all response tokens.
4. For every feature, **Cohen's d** compares wrong and correct problems: (mean on wrong − mean on correct) divided by
   the pooled standard deviation. The 10 features with the largest |d| are listed.

## How to read the feature tables

- **feature:** the feature's index in that SAE. For Llama Scope it links to its Neuronpedia dashboard (top activating
  examples from the SAE's training data and an automatic description).
- **Cohen's d:** **positive = more active on wrong answers**, negative = more active on correct answers. Roughly,
  |d| ≈ 0.2 is small, 0.5 medium and 0.8+ large. With 34 (math) or 12 (code) wrong examples, a few unusual problems
  can produce a large d by chance. Thousands of features are tested, so some large values are expected from noise alone.
- **mean act. (wrong / correct):** the feature's average activation in each group. 0 means it rarely fires.
- **peak tokens:** for the 5 problems where the feature fires hardest, the token at its single strongest activation.
  If these are all the same word (e.g. 'Wait'), the feature is likely tracking that word rather than a reasoning state.
- **strongest context:** the text around the single strongest activation, with the peak token in [[double brackets]]
  and ⏎ for newlines.

## Known confound for the response-mean summary

Wrong answers are often long or truncated: 17 of the 34 wrong GSM8K answers and 11 of the 12 wrong HumanEval+ answers
hit the 4096-token limit. Features that track response length, repetition or looping, or that fire mostly late in long
traces, will therefore separate wrong from correct without having anything to do with reasoning quality. The
last-prompt-token summary has no such confound, since it's measured before anything is generated.

## Reconstruction quality

Each SAE section ends with how well the SAE reproduces this model's real activations on these exact generations. If
reconstruction is poor, feature activations may not reflect what the model is actually computing.

