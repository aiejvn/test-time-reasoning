# SAE feature exploration: DeepSeek-R1-Distill-Llama-8B

**Status:** 3 of 5 SAEs finished; still queued: Llama Scope, layer 30, Galichin et al. paper SAE, layer 19. This file is regenerated after each SAE finishes.

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


## llamascope_l19: layer 19, 32,768 features

Llama Scope R1-Distill (OpenMOSS / fnlp): a sparse autoencoder trained on this model's layer output, on 400M tokens of general web text (SlimPajama) plus 400M tokens of R1-style math reasoning traces (OpenR1-Math-220k). Feature dashboards are on Neuronpedia.

### gsm8k_dev, pooled at the last prompt token (the `<think>\n` position, before any reasoning is generated)

n = 200 problems: 166 correct, 34 wrong.

| feature | Cohen's d | mean act. (wrong) | mean act. (correct) | peak tokens | strongest context |
|---|---|---|---|---|---|
| [10679](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/10679) | -0.87 | 8.226 | 9.317 | 'Please', 'Please', '⏎', 'Please', '⏎' |  prevent the pipes from freezing?⏎[[Please]] reason step by |
| [5612](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/5612) | -0.80 | 0.255 | 0.813 | ',', ',', ',', ',', ',' | boxed{}.<｜Assistant｜><think>⏎Okay[[,]] so I have |
| [8303](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/8303) | +0.66 | 9.502 | 8.935 | 'Erin', 'doesn', 'doesn', 'doesn', '624' |  that again. Let E be[[ Erin]]'s current points |
| [23964](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/23964) | -0.58 | 20.116 | 20.451 | '⏎', '⏎', '⏎', '⏎', '⏎' |  within \boxed{}.<｜Assistant｜><think>[[⏎]]Okay, so |
| [23556](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/23556) | +0.58 | 15.641 | 15.056 | 'mosaic', 'Kylie', 'big', 'Jenna', 'Ginger' | 24 mosaic tiles. How many[[ mosaic]] tiles would Mr |
| [25082](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/25082) | -0.54 | 4.449 | 4.994 | '⏎', '⏎', '⏎', 'Please', '⏎' |  within \boxed{}.<｜Assistant｜><think>[[⏎]]Okay, so |
| [25984](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/25984) | +0.49 | 7.566 | 7.154 | '⏎', '⏎', '⏎', '⏎', '⏎' |  within \boxed{}.<｜Assistant｜><think>[[⏎]]Okay, so |
| [24752](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/24752) | -0.47 | 0.159 | 0.357 | '⏎⏎', '⏎⏎', '⏎⏎', '⏎⏎', '⏎⏎' | {27} hours.⏎</think>[[⏎⏎]]First, we |
| [18337](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/18337) | +0.43 | 0.101 | 0.000 | '.', '.', '.', '.', '.' |  many degrees the second bottle is[[.⏎⏎]]Hmm, let |
| [16458](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/16458) | -0.41 | 0.497 | 0.820 | 'so', 'so', 'so', 'so', 'so' | {}.<｜Assistant｜><think>⏎Okay,[[ so]] I have this |

### gsm8k_dev, pooled at the mean over all response tokens (reasoning + answer)

n = 200 problems: 166 correct, 34 wrong.

| feature | Cohen's d | mean act. (wrong) | mean act. (correct) | peak tokens | strongest context |
|---|---|---|---|---|---|
| [9010](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/9010) | -1.83 | 0.095 | 0.393 | ':', '\', ':', ':', '7' |  multiply both sides by 4[[:⏎⏎]]\[⏎N |
| [31392](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/31392) | -1.82 | 0.007 | 0.023 | '-check', '-check', '-check', '-check', '-check' | 694.⏎⏎Let me just double[[-check]] my calculations to |
| [9187](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/9187) | -1.81 | 0.005 | 0.016 | 'boxed', 'boxed', 'boxed', 'boxed', 'boxed' |  Tanya uses is \(\[[boxed]]{4}\ |
| [19442](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/19442) | -1.80 | 0.014 | 0.037 | '.', '.', '.', '.', '.' |  $15 spent on lemons[[.⏎⏎]]**Final Answer |
| [24752](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/24752) | -1.74 | 0.009 | 0.023 | '⏎⏎', '⏎⏎', '⏎⏎', '⏎⏎', '⏎⏎' | {27} hours.⏎</think>[[⏎⏎]]First, we |
| [23335](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/23335) | -1.74 | 0.008 | 0.021 | '**', '**', '**', '**', '**' |  15.⏎⏎**Final Answer[[**⏎]]The average size |
| [18510](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/18510) | -1.73 | 0.009 | 0.024 | '</think>', '</think>', '</think>', '</think>', '</think>' |  is \boxed{110}.⏎[[</think>]]⏎⏎Rome |
| [20807](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/20807) | -1.73 | 0.006 | 0.016 | 'Okay', 'Okay', 'Okay', 'Okay', 'Okay' |  15 = 65⏎⏎[[Okay]], so now |
| [6647](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/6647) | -1.73 | 0.009 | 0.026 | '**', '**', '**', '**', '**' | -star rows is 8.⏎⏎[[**]]Final Answer**⏎ |
| [26755](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/26755) | -1.73 | 0.007 | 0.019 | 'Answer', 'Answer', 'Answer', 'Answer', 'Answer' |  is 12.⏎⏎**Final[[ Answer]]**⏎The total |

### humanevalplus_dev, pooled at the last prompt token (the `<think>\n` position, before any reasoning is generated)

n = 32 problems: 20 correct, 12 wrong.

| feature | Cohen's d | mean act. (wrong) | mean act. (correct) | peak tokens | strongest context |
|---|---|---|---|---|---|
| [23556](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/23556) | +1.37 | 21.272 | 20.745 | '30', 'sum', '5', 'sum', 'Examples' | :⏎    is_multiply_prime([[30]]) == True |
| [11036](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/11036) | +1.20 | 0.911 | 0.136 | '?', '.', '.', '.', '.' |  the correct way to model this[[?⏎⏎]]Wait, perhaps |
| [13133](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/13133) | -1.12 | 2.097 | 2.263 | 'Okay', 'Okay', 'Okay', 'Okay', 'Okay' |     """⏎'''<｜Assistant｜><think>⏎[[Okay]], I need |
| [8303](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/8303) | +0.99 | 17.727 | 17.245 | '⏎', '⏎', '⏎', '⏎', '⏎' | ]⏎    """⏎'''<｜Assistant｜><think>[[⏎]]Okay, I |
| [24963](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/24963) | +0.98 | 0.871 | 0.220 | '5', 'through', 'select', '-', 'True' | 3, 4, [[5]]) == True |
| [4219](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/4219) | +0.92 | 4.045 | 3.921 | '⏎', '⏎', '⏎', '⏎', '⏎' | '⏎    '''⏎'''<｜Assistant｜><think>[[⏎]]Okay, I |
| [30835](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/30835) | +0.91 | 1.797 | 1.675 | 'the', ' ', '⏎', '⏎', '⏎' |  says "three distinct elements in[[ the]] list". So |
| [22285](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/22285) | -0.86 | 1.709 | 2.079 | '⏎⏎', '⏎⏎', ')', '⏎⏎', '⏎⏎' | c]⏎    return min_path[[⏎⏎]]Wait, but |
| [21897](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/21897) | -0.79 | 1.165 | 1.581 | ':', '**', 'So', ':', 'have' |  function can be as simple as[[:⏎⏎]]return (a |
| [12996](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/12996) | +0.70 | 2.413 | 2.151 | 'doesn', 'lic', 'up', 'doesn', 'doesn' | ) is 2, which[[ doesn]]'t sum to |

### humanevalplus_dev, pooled at the mean over all response tokens (reasoning + answer)

n = 32 problems: 20 correct, 12 wrong.

| feature | Cohen's d | mean act. (wrong) | mean act. (correct) | peak tokens | strongest context |
|---|---|---|---|---|---|
| [21639](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/21639) | -3.13 | 0.063 | 0.353 | '):', '=', '):', '):', 'return' | :⏎⏎def compare(game, guess[[):⏎]]    return [ |
| [23081](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/23081) | -2.91 | 0.004 | 0.065 | 'by', 'is', 'exactly', 'equal', 'sorted' |  where the values at indices divisible[[ by]] three are sorted |
| [13214](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/13214) | -2.83 | 0.108 | 0.401 | '[k', '(c', 'a', '',', '(key' |  for k in counts if counts[[[k]]] == max |
| [19677](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/19677) | -2.79 | 0.000 | 0.003 | ''''', ''''', ''''', ''''', ''''' |  "Saturn")⏎    '''⏎[[''']]<｜Assistant｜><think>⏎ |
| [30880](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/30880) | -2.64 | 0.001 | 0.010 | '.', '.', '.', '.', '.' |  remaining stock minus the amount eaten[[.⏎⏎]]This approach efficiently |
| [12425](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/12425) | -2.53 | 0.004 | 0.032 | 'is', 'the', 'function', ',', 'is' | Wait, the initial code provided[[ is]] just the function |
| [14528](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/14528) | -2.52 | 0.001 | 0.004 | '<think>', '<think>', '<think>', '<think>', '<think>' | 5⏎    """⏎'''<｜Assistant｜>[[<think>]]⏎Okay, |
| [8376](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/8376) | -2.50 | 0.030 | 0.107 | ''', 'not', ''', 'i', 'code' |  'a' and 'b[[']] are added to |
| [4400](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/4400) | -2.50 | 0.027 | 0.117 | 'w', 'c', 'str', '(string', '(text' |  will_it_fly(q,[[ w]]):⏎    # |
| [1758](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/19-llamascope-slimpj-openr1-res-32k/1758) | -2.44 | 0.141 | 1.149 | '.', ';', 'is', ',', 'is' |  sequence with a single hyphen[[.]] If the length |

### Shared features between math and code

- prompt_last: 2 of the top 10 features appear in both gsm8k_dev and humanevalplus_dev (8303, 23556).
- resp_mean: 0 of the top 10 features appear in both gsm8k_dev and humanevalplus_dev.

### Reconstruction quality on our data

How well this SAE reproduces the model's actual layer output on these generations (all tokens except BOS). Cosine 1.0 and relative error 0.0 would be perfect. L0 is the average number of features active per token.

| benchmark | problems | mean cosine | mean relative error | mean L0 |
|---|---|---|---|---|
| gsm8k_dev | 200 | 0.852 | 0.274 | 56.9 |
| humanevalplus_dev | 32 | 0.816 | 0.333 | 66.6 |


## llamascope_l29: layer 29, 32,768 features

Llama Scope R1-Distill (OpenMOSS / fnlp): a sparse autoencoder trained on this model's layer output, on 400M tokens of general web text (SlimPajama) plus 400M tokens of R1-style math reasoning traces (OpenR1-Math-220k). Feature dashboards are on Neuronpedia.

### gsm8k_dev, pooled at the last prompt token (the `<think>\n` position, before any reasoning is generated)

n = 200 problems: 166 correct, 34 wrong.

| feature | Cohen's d | mean act. (wrong) | mean act. (correct) | peak tokens | strongest context |
|---|---|---|---|---|---|
| [16704](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/16704) | -0.88 | 0.918 | 1.409 | 'Okay', 'Okay', 'Okay', 'Okay', 'Okay' |  \boxed{}.<｜Assistant｜><think>⏎[[Okay]], so I |
| [8618](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/8618) | -0.79 | 0.694 | 1.330 | '⏎', '⏎', '⏎', '⏎', '⏎' |  within \boxed{}.<｜Assistant｜><think>[[⏎]]Okay, so |
| [10221](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/10221) | -0.77 | 1.686 | 2.331 | '⏎', '⏎', '⏎', '⏎', '⏎' |  within \boxed{}.<｜Assistant｜><think>[[⏎]]Okay, so |
| [20421](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/20421) | +0.73 | 4.508 | 3.954 | '⏎', '⏎', '⏎', '⏎', '⏎' |  within \boxed{}.<｜Assistant｜><think>[[⏎]]Okay, so |
| [22199](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/22199) | -0.69 | 13.157 | 13.887 | '⏎', '⏎', '⏎', '⏎', '⏎' |  within \boxed{}.<｜Assistant｜><think>[[⏎]]Okay, so |
| [9152](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/9152) | +0.67 | 3.289 | 2.894 | 'Erin', 'Mar', 'Jog', 'the', 'ck' |  that again. Let E be[[ Erin]]'s current points |
| [11742](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/11742) | -0.64 | 2.321 | 2.904 | '⏎', '⏎', '⏎', '⏎', '⏎' |  within \boxed{}.<｜Assistant｜><think>[[⏎]]Okay, so |
| [3668](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/3668) | +0.63 | 6.890 | 6.474 | '⏎', '⏎', '⏎', '⏎', '⏎' |  within \boxed{}.<｜Assistant｜><think>[[⏎]]Okay, so |
| [655](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/655) | +0.62 | 7.491 | 6.930 | '⏎', '⏎', '⏎', '⏎', '⏎' |  within \boxed{}.<｜Assistant｜><think>[[⏎]]Okay, so |
| [22422](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/22422) | -0.53 | 5.478 | 5.704 | ':', '"', '"', '"', '"' | .⏎⏎Plugging in the numbers[[:⏎⏎]]Total matches = |

### gsm8k_dev, pooled at the mean over all response tokens (reasoning + answer)

n = 200 problems: 166 correct, 34 wrong.

| feature | Cohen's d | mean act. (wrong) | mean act. (correct) | peak tokens | strongest context |
|---|---|---|---|---|---|
| [20931](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/20931) | -1.94 | 0.016 | 0.046 | 'Therefore', 'Therefore', 'Therefore', 'Therefore', 'Therefore' |  3 is 48.⏎⏎[[Therefore]], John consumed |
| [20316](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/20316) | -1.81 | 0.030 | 0.060 | 'Let', 'Let', 'Let', 'Let', 'Let' |  the rest must be children.⏎⏎[[Let]] me write that |
| [8144](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/8144) | -1.79 | 0.022 | 0.046 | 'model', 'compute', 'structure', 'simplify', 'calculate' | ounding.⏎⏎So, let's[[ model]] this. The |
| [10216](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/10216) | -1.78 | 0.004 | 0.014 | ']', ']', ']', '.', ']' |  = 20\%⏎\[[]⏎⏎]]Thus, the |
| [25101](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/25101) | -1.78 | 0.004 | 0.011 | 'Final', 'Final', 'Final', 'Final', 'Final' |  is 48 pieces.⏎⏎**[[Final]] Answer**⏎The |
| [29185](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/29185) | -1.78 | 0.012 | 0.034 | '}.', '.', '.', '.', '}.' |  plate is \boxed{8[[}.⏎]]</think>⏎⏎M |
| [6255](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/6255) | -1.77 | 0.005 | 0.014 | 'Final', 'Final', 'Final', 'Final', 'Final' |  answer is 16.⏎⏎**[[Final]] Answer**⏎The |
| [18520](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/18520) | -1.76 | 0.006 | 0.016 | 'double', 'double', 'double', 'double', 'double' | imeters.⏎⏎Wait, let me[[ double]]-check my calculations |
| [29780](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/29780) | -1.76 | 0.003 | 0.009 | '**', '**', '**', '**', '**' |  15.⏎⏎**Final Answer[[**⏎]]The average size |
| [2219](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/2219) | -1.75 | 0.008 | 0.022 | '{', '{', '{', '{', '{' |  Answer**⏎Jeff gets \boxed[[{]]80} dollars |

### humanevalplus_dev, pooled at the last prompt token (the `<think>\n` position, before any reasoning is generated)

n = 32 problems: 20 correct, 12 wrong.

| feature | Cohen's d | mean act. (wrong) | mean act. (correct) | peak tokens | strongest context |
|---|---|---|---|---|---|
| [19552](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/19552) | -1.14 | 2.173 | 2.304 | 'this', 'this', 'this', 'this', 'this' | But perhaps for the purposes of[[ this]] problem, it |
| [14568](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/14568) | -1.11 | 16.466 | 16.617 | ',', 'Both', ',', 'Or', 'the' | 29%24 is 5[[,]] no.⏎⏎i |
| [30548](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/30548) | -0.97 | 1.473 | 1.576 | '.', '.', '.', '".', '.' |  the actual score for each match[[.⏎⏎]]### Approach⏎ |
| [12372](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/12372) | +0.93 | 2.682 | 2.465 | 'bucket', 'carrots', 'lst', 'conson', '19' | ,1,1]]⏎           [[ bucket]]_capacity :  |
| [16207](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/16207) | +0.75 | 0.560 | 0.079 | '','', '²', '2', '=', '>' | ' → letters = ['a[[',']]b','c |
| [6601](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/6601) | -0.73 | 3.569 | 3.926 | '⏎', '⏎', '⏎', '⏎', '⏎' | ']⏎    """⏎'''<｜Assistant｜><think>[[⏎]]Okay, I |
| [22342](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/22342) | -0.72 | 0.509 | 0.943 | 'by', ''.', '(', 'row', 'in' |  multiple groups for nested parentheses separated[[ by]] spaces.⏎    |
| [22422](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/22422) | -0.69 | 2.480 | 2.678 | ']', '.', '.', ':', '⏎' | ,8,9,2[[]⏎⏎]]to_sort = |
| [3957](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/3957) | -0.68 | 1.245 | 1.530 | 'in', 'in', 'in', 'in', 'in' | ]⏎⏎Wait, but wait:[[ in]] the sample, |
| [27762](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/27762) | -0.66 | 0.510 | 1.010 | 'in', 'in', 'in', 'in', 'in' | , else a float.⏎⏎But[[ in]] Python,  |

### humanevalplus_dev, pooled at the mean over all response tokens (reasoning + answer)

n = 32 problems: 20 correct, 12 wrong.

| feature | Cohen's d | mean act. (wrong) | mean act. (correct) | peak tokens | strongest context |
|---|---|---|---|---|---|
| [13228](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/13228) | -3.72 | 0.006 | 0.034 | '.append', '.append', '.append', '.get', '.append' |  + 1):⏎        prefixes[[.append]](string[0 |
| [3444](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/3444) | -3.04 | 0.003 | 0.015 | 'def', 'def', 'def', 'def', 'def' | def fix_spaces(text):⏎   [[ def]] replace_spaces(match |
| [12560](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/12560) | -2.72 | 0.009 | 0.089 | 'of', 'is', 'planets', 'of', 'from' |  a given number is the product[[ of]] exactly three prime |
| [5817](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/5817) | -2.72 | 0.002 | 0.014 | '.', '.', '.', '.', '.' |  more spaces in the input string[[.⏎]]2. ** |
| [1350](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/1350) | -2.55 | 0.000 | 0.004 | 'perform', 'problem', 'this', 'this', '9' |  the solution is both elegant and[[ perform]]ant. |
| [7721](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/7721) | -2.50 | 0.000 | 0.004 | 'correct', 'it', 'approach', 'work', 'above' | .⏎⏎So the code should be[[ correct]].⏎</think>⏎⏎ |
| [28601](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/28601) | -2.45 | 0.001 | 0.006 | '⏎⏎', '⏎⏎', '⏎⏎', '⏎⏎', ')' |  max_count}⏎    return result[[⏎⏎]]Wait, but |
| [4238](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/4238) | -2.41 | 0.000 | 0.003 | ',', ',', ',', ',', ',' | </think>⏎⏎To solve this problem[[,]] we need to |
| [3003](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/3003) | -2.39 | 0.000 | 0.001 | '1', '1', '1', '1', '1' |  each row.⏎⏎### Approach⏎[[1]]. **Collect |
| [25707](https://www.neuronpedia.org/deepseek-r1-distill-llama-8b/29-llamascope-slimpj-openr1-res-32k/25707) | -2.39 | 0.003 | 0.009 | '2', '2', '2', '2', '2' |  if q is a palindrome.⏎[[2]]. If not |

### Shared features between math and code

- prompt_last: 1 of the top 10 features appear in both gsm8k_dev and humanevalplus_dev (22422).
- resp_mean: 0 of the top 10 features appear in both gsm8k_dev and humanevalplus_dev.

### Reconstruction quality on our data

How well this SAE reproduces the model's actual layer output on these generations (all tokens except BOS). Cosine 1.0 and relative error 0.0 would be perfect. L0 is the average number of features active per token.

| benchmark | problems | mean cosine | mean relative error | mean L0 |
|---|---|---|---|---|
| gsm8k_dev | 200 | 0.858 | 0.262 | 40.4 |
| humanevalplus_dev | 32 | 0.828 | 0.312 | 44.6 |

