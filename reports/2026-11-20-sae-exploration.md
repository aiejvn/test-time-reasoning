# SAE feature exploration: DeepSeek-R1-Distill-Llama-8B

**Status:** 1 of 5 SAEs finished; still queued: Llama Scope, layer 19, Llama Scope, layer 29, Llama Scope, layer 30, Galichin et al. paper SAE, layer 19. This file is regenerated after each SAE finishes.

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


## llamascope_l4: layer 4, 32,768 features

Llama Scope R1-Distill (OpenMOSS / fnlp): a sparse autoencoder trained on this model's layer output, on 400M tokens of general web text (SlimPajama) plus 400M tokens of R1-style math reasoning traces (OpenR1-Math-220k). Feature dashboards are on Neuronpedia.

### gsm8k_dev, pooled at the last prompt token (the `<think>\n` position, before any reasoning is generated)

n = 200 problems: 166 correct, 34 wrong.

| feature | Cohen's d | mean act. (wrong) | mean act. (correct) | peak tokens | strongest context |
|---|---|---|---|---|---|
| [2552](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/2552) | +0.62 | 0.607 | 0.258 | 'have', 'have', 'have', 'of', 'have' |  had gone, so he could[[ have]] counted the ones |
| [2190](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/2190) | -0.51 | 2.821 | 3.289 | '<think>', '<think>', '<think>', '<think>', '<think>' |  answer within \boxed{}.<｜Assistant｜>[[<think>]]⏎Okay, |
| [23603](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/23603) | -0.50 | 0.143 | 0.397 | 'again', 'again', 'again', 'cookies', 'question' |  about it. Let me try[[ again]].⏎⏎If one |
| [1964](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/1964) | -0.45 | 2.477 | 2.596 | '⏎', '⏎', '⏎', '⏎', '⏎' |  within \boxed{}.<｜Assistant｜><think>[[⏎]]Okay, so |
| [22776](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/22776) | +0.44 | 1.916 | 1.706 | 'I', 'I', 'I', 'I', 'I' | <｜Assistant｜><think>⏎Okay, so[[ I]] have this problem |
| [13985](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/13985) | -0.43 | 19.436 | 19.830 | '⏎⏎', '⏎⏎', '⏎⏎', '⏎⏎', '⏎⏎' |  \boxed{5}.⏎</think>[[⏎⏎]]Let \( B |
| [3086](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/3086) | +0.41 | 2.856 | 2.664 | 'Find', 'find', 'find', 'find', 'out' |  21% per year.[[ Find]] the price of |
| [1572](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/1572) | -0.41 | 0.000 | 0.083 | 'answer', 'answer', 'answer', 'answer', 'answer' |  step, and put your final[[ answer]] within \boxed |
| [4735](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/4735) | +0.41 | 0.165 | 0.024 | '⏎⏎', '⏎⏎', '⏎⏎', '⏎⏎', '⏎⏎' |  \boxed{5}.⏎</think>[[⏎⏎]]Let \( B |
| [8070](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/8070) | +0.38 | 0.187 | 0.032 | 'value', 'Please', 'Please', 'Please', 'Please' | , if I use the present[[ value]] formula:⏎⏎\[ |

### gsm8k_dev, pooled at the mean over all response tokens (reasoning + answer)

n = 200 problems: 166 correct, 34 wrong.

| feature | Cohen's d | mean act. (wrong) | mean act. (correct) | peak tokens | strongest context |
|---|---|---|---|---|---|
| [13874](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/13874) | -1.79 | 0.018 | 0.039 | 'Let', 'Let', 'Let', 'Let', 'Let' |  after all these events happen.[[ Let]] me try to |
| [5318](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/5318) | -1.77 | 0.007 | 0.020 | '{', '{', '{', '{', '{' | **Final Answer**⏎\boxed[[{]]14}⏎</think> |
| [17271](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/17271) | -1.75 | 0.002 | 0.005 | 'Final', 'Final', 'Final', 'Final', 'Final' |  answer is 12.⏎⏎**[[Final]] Answer**⏎The |
| [1711](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/1711) | -1.75 | 0.008 | 0.025 | 'boxed', 'boxed', 'boxed', 'boxed', 'boxed' | 68 degrees.⏎⏎\[⏎\[[boxed]]{68}⏎ |
| [14714](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/14714) | -1.75 | 0.008 | 0.022 | 'Answer', 'Answer', 'Answer', 'Answer', 'Answer' |  the correct answer.⏎⏎**Final[[ Answer]]**⏎Mary will |
| [7083](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/7083) | -1.74 | 0.007 | 0.021 | '4', '4', '4', '9', '0' | .⏎⏎\[⏎\boxed{[[4]]}⏎\] |
| [10788](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/10788) | -1.74 | 0.007 | 0.020 | '**', '**', '**', '**', '**' |  for Jeff.⏎⏎**Final Answer[[**⏎]]Jeff gets \ |
| [5821](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/5821) | -1.74 | 0.009 | 0.023 | 'Okay', 'Okay', 'Okay', 'Okay', 'Okay' |  \boxed{}.<｜Assistant｜><think>⏎[[Okay]], so Ben |
| [1243](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/1243) | -1.73 | 0.002 | 0.005 | '**', '**', '**', '**', '**' |  $120.⏎⏎**Final Answer[[**⏎]]Jack has \ |
| [22830](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/22830) | -1.73 | 0.006 | 0.013 | ',', ',', ',', ',', ',' | boxed{}.<｜Assistant｜><think>⏎Okay[[,]] so I have |

### humanevalplus_dev, pooled at the last prompt token (the `<think>\n` position, before any reasoning is generated)

n = 32 problems: 20 correct, 12 wrong.

| feature | Cohen's d | mean act. (wrong) | mean act. (correct) | peak tokens | strongest context |
|---|---|---|---|---|---|
| [5765](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/5765) | -0.73 | 0.153 | 0.671 | ')', ',', '',', ',', ')' |     total = sum(sum(row[[)]] for row in |
| [12397](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/12397) | +0.68 | 2.396 | 2.191 | '⏎⏎', '⏎⏎', '⏎⏎', '⏎⏎', '⏎⏎' | 's the correct implementation.⏎</think>[[⏎⏎]]To solve this |
| [23895](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/23895) | -0.67 | 2.060 | 2.217 | '⏎', '⏎', '⏎', '⏎', '⏎' | ### Solution Code⏎'''python[[⏎]]def eat(number |
| [30216](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/30216) | -0.63 | 1.254 | 1.501 | '>>>', ''''', ''''', ''''', ''''' |  wait, the sample says:⏎⏎[[>>>]] median([-10 |
| [22653](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/22653) | -0.57 | 0.000 | 0.217 | '⏎⏎', '⏎⏎', '⏎⏎', '⏎⏎', '⏎⏎' | 's write the function.⏎</think>[[⏎⏎]]To solve this |
| [17967](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/17967) | -0.55 | 2.011 | 2.137 | '⏎', '⏎', '⏎', '⏎', '⏎' | ### Solution Code⏎'''python[[⏎]]from typing import |
| [9713](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/9713) | +0.55 | 2.690 | 2.560 | '   ', '   ', '⏎', '⏎', '⏎' | python⏎⏎def solve(N):⏎[[   ]] """Given a |
| [30796](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/30796) | -0.54 | 1.427 | 1.537 | ''''', ''''', ''''', ''''', ''''' |  returned.⏎⏎### Solution Code⏎[[''']]python⏎def |
| [3481](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/3481) | -0.53 | 0.250 | 0.617 | '⏎⏎', 'as', '",', '.', '⏎⏎' | , code steps:⏎⏎import math[[⏎⏎]]def largest_prime |
| [13985](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/13985) | +0.52 | 16.448 | 16.101 | ',', ':', '-', ' ', ' ' | 1". For N=150[[,]] the sum is |

### humanevalplus_dev, pooled at the mean over all response tokens (reasoning + answer)

n = 32 problems: 20 correct, 12 wrong.

| feature | Cohen's d | mean act. (wrong) | mean act. (correct) | peak tokens | strongest context |
|---|---|---|---|---|---|
| [20090](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/20090) | -3.47 | 0.001 | 0.004 | ',', ',', ',', ',', ',' | </think>⏎⏎To solve this problem[[,]] we need to |
| [9464](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/9464) | -3.38 | 0.000 | 0.003 | '###', '###', '###', '###', '###' |  of the other two sides.⏎⏎[[###]] Approach⏎1 |
| [18334](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/18334) | -3.30 | 0.062 | 0.618 | 'lowercase', 'divisible', 'prime', 'ets', 'replace' |  a word, convert it to[[ lowercase]] and check if |
| [6232](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/6232) | -3.15 | 0.009 | 0.086 | 'underscores', 'indices', 'sum', 'last', 'conson' |  given string to replace spaces with[[ underscores]], but with |
| [6546](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/6546) | -2.85 | 0.000 | 0.001 | '###', '###', '###', '###', '###' |  of the other two sides.⏎⏎[[###]] Approach⏎1 |
| [32386](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/32386) | -2.68 | 0.003 | 0.014 | '(s', '(lst', '(array', '(game', '(string' | '''python⏎⏎def select_words[[(s]], n):⏎ |
| [17760](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/17760) | -2.64 | 0.005 | 0.022 | '(r', '['', '{'', ':', '[' |  length⏎    return re.sub[[(r]]' +', |
| [2817](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/2817) | -2.61 | 0.005 | 0.019 | '.', '.', '.', '.', '.' | .⏎⏎So the code is correct[[.⏎]]</think>⏎⏎To |
| [15146](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/15146) | -2.55 | 0.000 | 0.001 | '⏎', '⏎', '⏎', 'def', '⏎' | ### Solution Code⏎'''python[[⏎]]def select_words |
| [3963](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/4-llamascope-slimpj-openr1-res-32k/3963) | -2.54 | 0.000 | 0.004 | ''', ''', ''', ''', ''' | ) == 3⏎''[['⏎⏎]]### Explanation⏎ |

### Shared features between math and code

- prompt_last: 1 of the top 10 features appear in both gsm8k_dev and humanevalplus_dev (13985).
- resp_mean: 0 of the top 10 features appear in both gsm8k_dev and humanevalplus_dev.

### Reconstruction quality on our data

How well this SAE reproduces the model's actual layer output on these generations (all tokens except BOS). Cosine 1.0 and relative error 0.0 would be perfect. L0 is the average number of features active per token.

| benchmark | problems | mean cosine | mean relative error | mean L0 |
|---|---|---|---|---|
| gsm8k_dev | 200 | 0.903 | 0.184 | 30.6 |
| humanevalplus_dev | 32 | 0.879 | 0.226 | 31.6 |

