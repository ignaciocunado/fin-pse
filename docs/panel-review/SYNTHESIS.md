# Fin-PSE panel review: synthesis

Six expert panellists reviewed the repository and the research plan in two rounds: independent position papers (`round1/`), then rebuttals after reading each other (`round2/`). This file is the coordinator's summary. The brief they worked from is `BRIEF.md`.

| # | Panellist | Final verdict | Confidence |
|---|---|---|---|
| 01 | Positional/structural encodings and expressivity | Continue with major changes | medium |
| 02 | Financial-crime graph ML | Continue with major changes, gated | medium-high |
| 03 | Temporal graphs | Continue with major changes, gated | medium |
| 04 | Graph self-supervised learning | Pivot | high |
| 05 | Empirical rigour (code audit) | Continue with major changes, gated | medium |
| 06 | Scalable and geometric graph learning | Continue with major changes, leaning pivot | medium |

## Verdict

**Not a dead end, but the project as currently designed should not be run.** Nobody voted to continue as planned and nobody voted dead end. The five "major changes" votes replace the targets, the encoder, the inputs, the backbone and the evaluation; panellist 04 calls the same set of changes a pivot, and that disagreement is about the label more than the substance.

All six put the probability that **feeding the exact 12 statistics as node features matches or beats frozen Fin-PSE at 0.80 to 0.85**. That is the control the plan itself names as decisive, so the expected outcome is that H1 and H2 fail in their current form. Every "continue" vote is conditional on a gate experiment (below).

What survives is the question: whether a learned, transferable structural encoder for directed, temporal transaction multigraphs can beat exact feature engineering. The panel considers that open and worth one redesigned attempt.

## What the panel agreed on

**Strengths**

- The question is real and appears unanswered on this benchmark.
- The plan already lists the right controls (handcrafted features, untrained encoder, capacity-matched baselines). They are not implemented yet.
- Chronological splits, a real out-of-simulator target (Ethereum), and a small, readable codebase.

**Scientific weaknesses**

1. **The encoder adds no information the backbone lacks.** Downstream it runs on the same sampled mini-batch, at the same depth and direction, with all-ones node inputs and fewer edge features than the backbone. It sits inside the backbone's own function class, so any gain would be an optimisation or capacity effect.
2. **All 12 targets are one-hop aggregates** of a node's incident edges (including the 2-cycle count; `flow_imbalance` is a function of two others). The ETL computes them exactly in one scan. Pretraining a network to approximate them is distillation of a cheap extractor into an expensive one.
3. **Six of the 12 targets are unobservable.** Sampling follows in-edges only and messages flow source to target, so out-degree, fan-out, out-volume, both outgoing gap statistics and flow imbalance are outside the seed node's computational graph.
4. **Time is not handled natively.** It enters as an absolute timestamp, z-scored per 2-day window in pretraining but per training split downstream, so the unit shifts by a few times on AML and by hundreds of times on Ethereum. No target depends on event order across nodes, and samplers are time-agnostic.
5. **The edge "encoding" is a frozen linear map of two raw features**, so it contributes nothing to edge classification.
6. **H4 (efficiency) is ill-posed.** The frozen encoder is re-run on every batch of every epoch to approximate statistics that a group-by computes exactly. The auditor puts the encoder cost on the Large graphs several orders of magnitude above the exact scan (a back-of-envelope estimate, not a measurement).
7. **The backbone is a weak baseline.** Plain directed GIN/PNA without reverse message passing, port numbering or ego IDs is well below the published adaptations for this benchmark, so a gain over it may only show the encoder supplying out-direction information.

**Defects that block any valid result today** (auditor, panellist 05; details and line references in `round1/05-empirical-rigor.md`)

- The with-encodings arm cannot run: every `_encodings` config has an empty `encodings_file`, both pretraining job scripts reference a config that does not exist, and the ETL writes to a different directory than the loader reads.
- PR-AUC is computed on the raw class-1 logit instead of a probability or logit difference. In a synthetic check the same model scored 0.004 by the code's method and 0.277 by the correct one.
- Seed edges that the neighbour sampler does not pick are silently dropped from the metrics, so the scored test set is a sampler-dependent subset.
- Edge ids are stored as float32, which collides above about 16.7 million edges and corrupts the seed mask on the Large tables.
- The "frozen" encoder's BatchNorm returns to training mode and updates on downstream data.
- Random node features are drawn once and cached, are one-dimensional, and are replaced by all-ones downstream.
- Weight decay is never applied under `adam`; no final test number is persisted; nothing aggregates across seeds.
- The with-encodings arm has about a third more trainable parameters and half the own-embedding width, so the comparison is not capacity-matched.

The auditor ran synthetic checks for message direction, BatchNorm mode, the metric and parameter counts. Sampler behaviour was verified by reading the library source only, on a newer library version than the project pins. No training was run by anyone; the data is not in the repository.

## Where the panel disagreed

- **Does swapping in harder targets rescue pretrain-then-freeze?** 01, 05 and 06 say yes if the targets are expensive and outside the backbone's function class (time-respecting cycles, scatter-gather, walk profiles). 04 says no: it is still distillation with the exact miner as a ceiling, and a learned encoder can only win with a non-deterministic objective (masked or future-event prediction) or in low-label and cross-ledger settings. 01 replies that changing the objective does not fix containment; architecture and inputs must change first.
- **Causal or retrospective evaluation?** 03 and 05 want time-respecting sampling and statistics computed as of each transaction's time. 02 argues AML monitoring is retrospective and the onward transfer after a suspicious credit is the evidence of layering. Resolution offered: report both.
- **Precompute encodings once per split?** 01, 05 and 06 recommend it for cost. 03 objects that on a temporal edge task this scores an early transaction with later information and gives every transaction of an account the same vector.
- **Which backbone for the gate?** 02 wants the adapted backbone first. 03 and 05 want both, because on the plain backbone the exact-statistics control wins for a reason unrelated to pretraining (it supplies out-direction information), while on the adapted backbone possibly nothing helps.
- **Is there structural headroom at all?** 01 and 06 both ask whether exact multi-hop features help on top of a backbone with reverse message passing, ports and ego IDs. If they do not, every encoding direction is moot on this benchmark.

## Recommended next steps

Merged from the six ballots, in order. Steps 1 to 3 were on every ballot.

1. **Repair the harness on the Small datasets.** Checkpoint path and config/script consistency; integer edge ids in a separate tensor; score every seed edge; PR-AUC on a probability; encoder locked in eval mode; one input convention between pretraining and downstream; per-seed results file.
2. **Pretext audit (about a day).** Held-out per-target R² for the pretrained encoder, and a ridge regression of the out-direction targets on the in-direction targets. This shows whether the encoder learned anything beyond correlations.
3. **Gate experiment.** HI-Small and LI-Small, 5 to 10 seeds, on both the plain backbone and one with reverse message passing. Arms: no encoding; exact 12 statistics; random frozen encoder; frozen Fin-PSE; a depth-matched backbone. Add a gradient-boosted-tree baseline on mined graph features, and run at a low-label setting (about 5%) as well as full labels.
   - **Kill criterion:** if frozen Fin-PSE does not beat both the exact statistics and the random encoder on the reverse-message-passing backbone, drop the current design.
4. **Headroom test.** Exact multi-hop typology features (temporal cycles, scatter-gather) on the adapted backbone, with per-typology recall. If these do not help, stop pursuing encodings on this benchmark.
5. **Only if step 4 shows headroom, rebuild.** Reverse edges with a direction flag; relative log time and log amounts; multi-aggregator, identity-aware encoder; targets that are expensive and order-dependent (the 2-step flow entropy already computed in the ETL is the cheapest start); a label-scarcity curve and a second generator before any transfer claim.

Drop H4 as stated. An efficiency claim only makes sense against features that are expensive to compute exactly.

## Alternative research directions

Ranked by how many of the six ballots placed them in the top three.

1. **Time-respecting walk and motif encodings, training-free or sketched (5 of 6).** Estimate return probabilities, walk profiles and temporal-motif counts under the constraint that each step is later than the previous one, using random probes or sketches instead of pretraining. It targets the actual gap (direction and event order), is near-linear, and can be falsified without a pretraining pipeline. Risk: variance dominated by hub accounts. First experiment: exact temporal 3- and 4-cycle and scatter-gather counts as features on HI-Small, then replace the exact counts with a sketch and measure the loss.
2. **Message passing on the transaction line graph (4 of 6).** Make transactions the nodes and link A→B at t1 to B→C at t2 only when 0 < t2 − t1 ≤ δ, so every message follows a feasible money path. Three panellists reached this independently. Risk: blow-up at hub accounts. First experiment: a 2-layer model on HI-Small with capped fan-out, compared against the reverse-message-passing backbone.
3. **A leakage-audited benchmark with cross-generator transfer (3 of 6).** Full-split scoring, correct metrics, causal and retrospective protocols, strong feature baselines, and transfer across simulators and to real ledgers. The defects found here are inherited from common practice on this benchmark, so published numbers are probably not comparable. Lowest scientific risk, and it produces the baseline every other direction must beat.
4. **Amortised temporal-motif encoder (2 of 6, contested).** The one version of the original idea where a learned encoder has a cost argument, because exact temporal-motif mining is expensive. Panellist 04 advises against it as still being distillation.
5. **Hodge / edge-flow encodings (2 of 6).** Treat transactions as an edge flow and split it into gradient and circular components with one sparse least-squares solve per window. The only genuinely global, positional proposal. Its own proposer conceded it ignores event order and largely smooths flow imbalance; best used as an extra edge feature.
6. **Multi-ledger masked-event pretraining (1 of 6, contested).** Mask or predict future events across several ledgers and evaluate leave-one-ledger-out with linear probes and few labels, as a step towards a transaction-graph foundation model. The auditor advises against starting here, since it multiplies data, compute and evaluation surface before one trustworthy number exists.

Advised against by four of six: in-context relational or tabular foundation models as the main line. At about 0.1% prevalence a context of a few thousand rows holds two or three positives. It is fine as a single baseline row.

## Caveats

- No experiment was run. Every performance statement is a prediction.
- Literature checks were partial: several panellists could not reach the preprint server, and older citations and published scores in the papers are marked as from memory. Verify them before citing.
- Cost figures are back-of-envelope estimates.
