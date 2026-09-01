# Mechanistic Interpretability of Test-Time Reasoning in Large Language Models

### A Sparse-Autoencoder Account of Uncertainty, Commitment, and Error Recovery

**Research Proposal**

Proposed duration: 9 months (July 2026 – March 2027)

**Project Leads:** Rastin Aghighi, School of Computing, Queen's University; Kevin Wang, School of Computing, Queen's University

**Proposed Faculty Advisor:** TBD

*June 2026*

---

## Abstract

Modern reasoning models solve multi-step problems by generating extended chains of thought at inference time, yet we still lack a mechanistic account of how they reason and, more urgently, of why they fail. The dominant failure mode of these models is distinctive and dangerous: confident, fluent, internally coherent reasoning that arrives at the wrong answer. Existing work characterizes reasoning only from the outside. Process-reward models score steps, entropy gates decide where to branch, and fine-tuning induces self-refinement; in every case, an external signal manages a behavior whose internal basis is unknown. In parallel, sparse autoencoders (SAEs) and attribution graphs have made it possible to decompose a model's internal activations into interpretable features and to trace those features into circuits, but only for static content, never for reasoning trajectories. We propose to connect these two lines. We will train SAEs on the intermediate-layer activations of open reasoning models while they reason, identify, and causally validate the features that govern key reasoning behaviors, uncertainty and branching, premise commitment, and error recovery, and characterize the mechanistic signature of miscalibrated confidence: the confident-but-wrong failures that surface-level signals such as token entropy cannot detect. Over nine months, a seven-member QMIND team will deliver an open SAE-based interpretability toolkit for reasoning models, a methodology paper establishing the approach, and a submission to a top venue on the strongest finding. The project sits directly at the intersection of test-time reasoning and model reliability questions the proposed advisor's group studies behaviorally and asks what those external signals correspond to inside the model.

## 1. Introduction and Motivation

Reasoning-specialized language models such as DeepSeek-R1, the OpenAI o-series, and the Qwen-Math family have shown that allocating compute at inference time, letting a model "think" in a long chain of intermediate steps before answering, substantially improves performance on mathematics, logic, and multi-hop problems. As these models move toward deployment in high-stakes settings, two gaps become pressing.

First, we do not understand the mechanism of reasoning. We can observe that a model branches, backtracks, or corrects itself from the outside by reading its output tokens, but we have no account of the internal computation that produces these behaviors. We cannot say what representation marks the moment a model commits to a premise, what distinguishes productive exploration from unproductive flailing, or what fires when a model recovers from an error rather than compounding it.

Second, reasoning models fail characteristically and dangerously. Their most consequential errors are not hesitant but confident: a fluent, well-structured chain that asserts a false intermediate result and rationalizes everything downstream from it. Reliability, the property that determines whether a model can be trusted in deployment, hinges on our ability to detect and control exactly this failure. Notably, surface-level uncertainty signals such as token entropy are blind to it by construction: when a model is confidently wrong, its output distribution is sharp, not flat.

Two research lines bear on these gaps, and they have developed in isolation. The behavioral line studies reasoning from the outside and manages it with external signals: automatic process supervision and step-level reward modelling (Rizvi et al., 2026), uncertainty-gated branching at test time (Li et al., 2026), within-inference self-refinement induced by fine-tuning (Puerto et al., 2025), and the graph structure of multi-step proofs (Zheng et al., 2024). The mechanistic line decomposes a model's internals into interpretable units: superposition explains why individual neurons are not the right unit of analysis (Elhage et al., 2022); sparse dictionary learning recovers interpretable features from real models (Bricken et al., 2023) and scales to production systems (Templeton et al., 2024); and attribution graphs trace those features into circuits (Ameisen et al., 2025; Lindsey et al., 2025). Crucially, the mechanistic line has so far been applied to static content features for entities, topics, and code, and not to the dynamics of a reasoning trajectory.

Our thesis is simple: the external signals the behavioral line relies on — step-correctness, uncertainty, and the impulse to refine — have internal representations that can be recovered and causally manipulated. We propose to find them. The instruments to look inside now exist; the behaviors worth looking at have been characterized; no one has pointed the former at the latter. This is the same phenomenon studied from the opposite vantage point, from the outside in, and it is the natural complement to the proposed advisor's program, which manages reasoning and probes model confidence from the behavioral side.

This proposal makes the following contributions:

- The first SAE-based decomposition of reasoning-process features, not static content features, in open reasoning models, together with a rigorous causal-validation protocol.
- A mechanistic characterization of miscalibrated confidence: the confident-but-wrong regime that entropy-based and other surface methods cannot detect.
- Causal-intervention experiments testing whether steering identified features measurably improves self-correction and accuracy.
- A cross-model study of how invariant these features are across model families, scale, and training regime (reinforcement-learned vs. distilled/supervised).
- An open-source interpretability toolkit and a diagnostic benchmark for reasoning models, released to the community.

## 2. Background and Related Work

### 2.1 Interpretability instruments

**Superposition and the case for dictionary learning.** Elhage et al. (2022) show in fully understood toy models that networks represent more features than they have neurons by encoding them as overlapping, non-orthogonal directions, so individual neurons are polysemantic and cannot be read directly. Whether a feature is stored, given a dedicated dimension, or dropped is governed by a phase change in its sparsity and importance. This is the formal reason our analysis cannot operate on neurons and must recover an overcomplete set of feature directions.

**Sparse autoencoders.** Bricken et al. (2023) demonstrate that an SAE trained on a real transformer's activations recovers monosemantic, causally meaningful features, and establishes four criteria for a genuine feature: high specificity, high sensitivity, an appropriate causal effect under intervention, and non-correspondence to any single neuron, along with the phenomena of feature splitting and universality. Templeton et al. (2024) scale the method to a production model (Claude 3 Sonnet), using scaling laws to choose dictionary size and recovering safety-relevant, steerable features. Together, these establish both the recipe we will follow and the evidence that it is tractable at the 7–8B scale we target.

**Circuit tracing.** Ameisen et al. (2025) and Lindsey et al. (2025) introduce cross-layer transcoders and attribution graphs to trace how features combine across layers into circuits, with case studies on multi-step reasoning, planning, hallucination, and chain-of-thought (un)faithfulness, including evidence that a model's stated reasoning can diverge from its actual computation. This is the most direct precedent for the circuit-level component of our work and our designated fallback when single features are insufficient.

### 2.2 Reasoning targets: the behavioral account

The behaviors we aim to explain mechanistically have been characterized behaviorally by the proposed advisor's group and collaborators, which both define our targets and ground the project in the advisor's research program.

- **Process supervision (SPARE; Rizvi, Zhu, and Gurevych, 2026).** A single-pass method that annotates the correctness of each reasoning step against a reference solution, matching the quality of search-based annotation at a fraction of the cost. It provides exactly the step-level correctness signal we need to label correct vs. incorrect steps for analysis.
- **Entropy-gated branching (Li, Callanan, Ghassel, and Zhu, 2026).** Branching only at high-uncertainty steps, with the key empirical finding that uncertainty spikes align with the model's mistakes. This operationalizes "the model is uncertain" from the output distribution and locates the decisive moments in a trace while remaining blind to the confident-but-wrong case, which motivates our RQ2.
- **Within-inference refinement (DCoT; Puerto, Chubakov, Zhu, Tayyar Madabushi, and Gurevych, 2025).** Fine-tuning a model to generate diverse reasoning chains sequentially in one pass so that it refines later chains based on earlier ones, achieving self-correction without external feedback. This yields a controlled pair of models with and without the refinement capability, ideal for isolating what the capability changes internally.
- **Reasoning structure (Zheng, Malon, Min, and Zhu, 2024).** Evidence that multi-step reasoning is graph-structured rather than linear, and that non-sequential (branching) reasoning is harder for all models. This supplies the structural vocabulary for our trajectory analysis and a sequential vs. non-sequential contrast we can correlate against internal features.

### 2.3 The gap we address

No prior work applies the interpretability instruments of Section 2.1 to the reasoning targets of Section 2.2. The behavioral line never looks inside the model; the interpretability line has not examined reasoning trajectories. This project bridges the two: it treats the external signals (step-correctness, entropy, the refinement impulse) as instrumentation for locating internal representations, and asks what those representations are, whether they are causal, and how far they generalize.

## 3. Research Questions

**RQ1 — Feature discovery.** What interpretable features are active during the key reasoning behaviors branching at uncertainty, premise commitment, backtracking, and error recovery, and how can each be causally validated rather than merely correlated?

**RQ2 — Miscalibrated confidence.** What is the mechanistic signature of confident-but-wrong reasoning, the low-entropy failures that surface signals cannot detect? Is there an internal confidence or commitment representation that fires when it should not, and does it predict these failures better than output-distribution statistics?

**RQ3 — Invariance.** How invariant are these features across model families (Qwen-2.5-Math vs. DeepSeek-R1-Distill-Llama), across scale, and across training regime (reinforcement-learned vs. distilled/supervised)?

**RQ4 — Intervention.** Can targeted interventions on the identified features, amplifying a self-correction feature, or suppressing a premise-commitment feature, steer reasoning toward self-correction and improved accuracy in a predictable way?

## 4. Methodology

### 4.1 Models

We study Qwen-2.5-Math-7B and DeepSeek-R1-Distill-Llama-8B: open, reasoning-specialized, and small enough to be tractable on available compute, while differing in training regime (the latter distilled from a reinforcement-learned teacher) so that RQ3 is addressable within the same study. The choice of a quantitative-reasoning model also connects the reliability question to the domain of robust mathematical and financial reasoning.

### 4.2 SAE training

Following the production-validated recipe of Templeton et al. (2024), we train sparse autoencoders on the residual stream at intermediate layers, capturing activations across full reasoning chains rather than at a single position. We use scaling laws to select dictionary size and activation-data budget rather than choosing them by trial and error, and we treat dictionary size as a deliberate experimental parameter, anticipating the feature splitting documented by Bricken et al. (2023). Implementation uses SAELens and TransformerLens, with experiment tracking in Weights & Biases.

### 4.3 Feature validation

Every feature we claim must pass the four-criteria protocol of Bricken et al. (2023): high specificity (the feature is active in the target context), high sensitivity (the context reliably activates it), an appropriate downstream causal effect under intervention, and non-correspondence to any single neuron. This rubric is built into the pipeline from the outset so that central claims are defensible rather than anecdotal.

### 4.4 Locating reasoning events

To analyze the few moments that matter without exhaustively examining every token, we use two external signals as instrumentation: token-level entropy and varentropy (Li et al., 2026) to flag uncertainty and branching points, and single-pass step-correctness annotation (Rizvi et al., 2026) to label correct vs. incorrect steps. We then compare the SAE features active at flagged versus unflagged steps, and at correct versus incorrect steps.

### 4.5 Causal interventions

We validate and test features using activation ablation, amplification (clamping a feature on or off), and cross-prompt activation patching, the steering paradigm established by Templeton et al. (2024). For RQ4 specifically, we test whether amplifying a candidate self-correction feature or suppressing a candidate premise-commitment feature changes reasoning outcomes in the predicted direction.

### 4.6 Circuit-level analysis (fallback)

Where a single feature does not suffice to explain a behavior, we apply attribution graphs and cross-layer transcoders (Ameisen et al., 2025), adapting the open Circuit Tracer tooling, which already supports comparable open models. Because this method is more computationally intensive, we budget it as a secondary instrument reserved for the behaviors that resist a single-feature account.

### 4.7 Evaluation

We quantify results on GSM8K and ARC-AGI 3, supplemented by purpose-built diagnostic sets that isolate branching, commitment, and recovery. We follow the experimental norms of the target literature: strong baselines (for RQ2, entropy-only detection of errors is the baseline our internal-feature detector must beat), ablations over dictionary size and layer choice, and statistical-significance testing of intervention effects.

## 5. Evaluation and Success Criteria

We define tiered success criteria so that the project yields a credible result even if its most ambitious aims are not reached, while leaving clear room for a high-impact finding.

- **Floor.** A working SAE pipeline on at least one model and a set of reasoning-process features that pass the four-criteria validation protocol, written up as a methodology paper. This is achievable by a single workstream and anchors the project.
- **Target.** A characterized phenomenon, for example, a confidence or commitment feature that predicts confident-but-wrong errors better than token entropy, or a self-correction feature whose amplification measurably improves accuracy.
- **Stretch.** A circuit-level account of one reasoning behavior, or a feature-level intervention that improves reliability consistently across both models, the basis for the top-venue submission.

## 6. Timeline and Milestones

The project runs for nine months, from July 2026 to March 2027. Staffing is deliberately matched to the calendar: the summer months are lighter and are used for engineering-heavy infrastructure work that does not require the full team, with the team at full strength once the academic terms begin in September.

| Phase | Months | Focus and deliverables |
|---|---|---|
| Phase 0 | Jul–Aug (1–2) | Infrastructure and SAE pipeline (lighter summer staffing). Train SAEs on Qwen-2.5-Math-7B; reproduce clean features; build the four-criteria validation harness and the entropy / step-correctness instrumentation. Deliverable: working pipeline and validated baseline features. |
| Phase 1 | Sep–Nov (3–5) | Feature discovery and validation (RQ1); begin confidence study (RQ2). Full team. Deliverable: validated reasoning-process feature set; methodology workshop paper drafted and submitted. |
| Phase 2 | Dec–Jan (6–7) | Causal interventions and confident-but-wrong analysis (RQ2, RQ4). Deliverable: intervention results and mechanistic confidence findings. |
| Phase 3 | Feb–Mar (8–9) | Cross-model generalization (RQ3) and synthesis. Replicate on DeepSeek-R1-Distill-Llama-8B; consolidate the strongest result. Deliverable: top-venue submission (via ACL Rolling Review). |

Publication targets: a methodology workshop paper around month five, establishing the SAE-on-reasoning approach and initial validated features, and a submission to a top venue by month nine on the strongest result, routed through ACL Rolling Review.

## 7. Team, Roles, and Training

The project is carried out by a five-member QMIND team led by one project manager, who sets research direction, manages milestones, and leads writing, supported by a research supervisor. The remaining members are organized into the project's workstreams: SAE training and infrastructure, feature validation, causal interventions, and evaluation and diagnostics.

**Training value.** The project trains undergraduate researchers in mechanistic interpretability and AI reliability, a methodology and a problem area that remain underrepresented at the undergraduate level in Canada, producing highly qualified personnel with skills directly relevant to both academic research and industry.

**Supervision.** Faculty advisor: TBD, for direction on the reasoning side and for venue strategy. To complement the advisor's expertise, we are securing an interpretability co-advisor from among Vector-affiliated and external interpretability researchers to support the methods side; this division of supervision is explicit in our plan rather than assumed.

## 8. Resources and Feasibility

- **Compute.** Compute is the project's main resource requirement. The 7–8B model scale is modest, and we plan to secure GPU resources through cloud providers such as AWS and similar cloud GPU services, in addition to any other resources we are able to secure; exact provisioning remains TBD pending available funding and partnerships.
- **Tooling.** SAELens, TransformerLens, HuggingFace, the open Circuit Tracer library, and Weights & Biases, all open and mature, mean we adapt existing infrastructure rather than build it from scratch.
- **Data.** Open benchmarks (GSM8K, ARC-AGI), with step-level annotations produced by the token-efficient single-pass method of Rizvi et al. (2026).
- **Risk management.** Tiered success criteria help ensure a shippable result; the four-criteria rubric guards against spurious features; and the circuit-tracing fallback covers behaviors that resist a single-feature account.

## 9. Expected Contributions and Broader Impact

**Scientific.** The first mechanistic account of reasoning-process features and of miscalibrated confidence in open reasoning models moves the study of test-time reasoning from a behavioral to a mechanistic footing.

**Practical.** An open toolkit and diagnostic benchmark the community can build on, and a candidate internal signal that could improve test-time reasoning methods, for instance, by gating branching on a genuine internal uncertainty feature rather than on token entropy.

**Reliability and safety.** A step toward mechanism, rather than bolt-on, control of reasoning failures, directly relevant to the reliability and model-confidence questions the proposed advisor's group studies.

## References

Ameisen, E., Lindsey, J., Pearce, A., et al. (2025). Circuit Tracing: Revealing Computational Graphs in Language Models. *Transformer Circuits Thread.*

Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., et al. (2023). Towards Monosemanticity: Decomposing Language Models with Dictionary Learning. *Transformer Circuits Thread.*

Cunningham, H., Ewart, A., Riggs, L., Huben, R., & Sharkey, L. (2023). Sparse Autoencoders Find Highly Interpretable Features in Language Models. arXiv:2309.08600.

DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.

Elhage, N., Hume, T., Olsson, C., Schiefer, N., et al. (2022). Toy Models of Superposition. *Transformer Circuits Thread.* arXiv:2209.10652.

Li, X., Callanan, E., Ghassel, A., & Zhu, X. (2026). Entropy-Gated Branching for Efficient Test-Time Reasoning. In *Proceedings of EACL 2026* (pp. 5054–5069). arXiv:2503.21961.

Lindsey, J., Gurnee, W., Ameisen, E., et al. (2025). On the Biology of a Large Language Model. *Transformer Circuits Thread.*

Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., & Mueller, A. (2024). Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models. arXiv:2403.19647.

Puerto, H., Chubakov, T., Zhu, X., Tayyar Madabushi, H., & Gurevych, I. (2025). Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs. In *Proceedings of ACL 2025* (pp. 3789–3808).

Rizvi, M. I. H., Zhu, X., & Gurevych, I. (2026). SPARE: Single-Pass Annotation with Reference-Guided Evaluation for Automatic Process Supervision and Reward Modelling. In *Proceedings of AAAI 2026* (pp. 32808–32816). arXiv:2506.15498.

Templeton, A., Conerly, T., Marcus, J., Lindsey, J., et al. (2024). Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet. *Transformer Circuits Thread.*

Wei, J., Wang, X., Schuurmans, D., Bosma, M., et al. (2022). Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In *Advances in Neural Information Processing Systems 35.*

Zheng, Z., Malon, C., Min, M. R., & Zhu, X. (2024). Exploring the Role of Reasoning Structures for Constructing Proofs in Multi-Step Natural Language Reasoning with Large Language Models. In *Proceedings of EMNLP 2024* (pp. 15299–15312).