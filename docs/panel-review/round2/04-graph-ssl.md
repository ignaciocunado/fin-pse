# Rebuttal 04: graph self-supervised learning and foundation models

## Where I now agree

- **Empirical rigour (05): nothing can run yet.** Every `_encodings` config crashes on an empty checkpoint path, PR-AUC ranks by a raw logit, and seed edges into hubs are dropped from the test metric. My five-arm experiment silently assumed a working pipeline. The fix list in 05 comes first.
- **Financial-crime ML (02): my control arms used a strawman backbone.** I proposed plain GIN and PNA. A gain there would be read as rediscovering reverse message passing. The gate has to include Multi-GIN with edge updates, and the Graph Feature Preprocessor with gradient boosting as the feature comparator.
- **Scalable geometric (06) and PSE theory (01): I was too generous on H4 and on the edge branch.** I wrote that H4 holds against the wrong comparator. Because the encoder runs inside every sampled batch, 06 argues it is not cheap against anything practical. I also missed that the edge "encoding" is a frozen rank-2 linear map.
- **Temporal graphs (03): better input design than mine.** Time relative to the seed edge, log-scaled with fixed multi-scale frequencies and no z-scoring, is cleaner than my "gap to previous event" proposal. 03's predictive pretraining (input window, future-window targets) is the same idea as my next-window head, better specified.
- **PSE theory (01): reciprocity is not usable downstream.** I called `r_2cycle` partly observable. Without node identity at inference, whatever was learned for it cannot be computed.
- **My own alternative 3 (amortised motif encoder) should be downgraded.** See the first disagreement.

## Where I disagree

**1. With 01, 05, 06 (and 02's improvement 4): swapping in "expensive" targets does not rescue pretrain-then-freeze.** Four papers propose keeping statistic regression but regressing cycle counts, scatter-gather counts or walk profiles instead. That is still distillation, and a student's ceiling is its teacher. The panel's own evidence says the teacher is cheap: 02 cites exact streaming pattern mining as a deployed CPU baseline, 06's cost table puts exact features orders of magnitude below any encoder pass, and 01 lists "the exact miner is already fast" as the risk. 01's oracle test concedes the structure: if exact targets win as features, the project "reduces to whether a learned encoder recovers most of that gap at lower cost". There is little cost left to recover, and a sampled MPNN cannot count cycles beyond length four without per-seed ego IDs, which removes the cost advantage entirely. A learned encoder can only beat exact features when its objective is not a deterministic function of its input (masked or future-event prediction), or when the test is label-scarce or cross-ledger. Regression to any hand-computed statistic fails that condition.

**2. With 05: the proposed gate tests the regime where pretraining is least likely to matter.** Ten seeds, seven arms, per-arm tuning and Holm correction, all at full labels, is an expensive way to measure a result five of six panellists already predict. It also has no label-budget axis. 02 and I both argue that frozen pretrained features plausibly win only at 1 to 5% of labels, which is also the realistic setting. A three-arm gate at two label budgets and five seeds is cheaper and more informative. If the pretrained arm loses at 5% labels too, the objective is finished.

**3. With 01: expressivity is not the binding constraint.** 01 would rebuild the encoder (resampled random features, depth 6 to 8) so that it exceeds the backbone's function class. On a fully labelled benchmark, the supervised route to the same expressivity already exists (reverse message passing, ports, ego IDs). 02's third outcome, "nothing helps on top of Multi-GIN+EU", is the likely result of that effort. Pretraining earns its place through label efficiency and transfer, and neither is tested by an expressivity argument.

**4. On the verdicts.** Five papers say CONTINUE WITH MAJOR CHANGES, then replace the targets, the loss, the encoder architecture, the input parameterisation, the backbone and the evaluation. What survives is the ETL and the question. I call that a pivot. The label matters because "continue" keeps the Fin-PSE main table as the deliverable.

## Open question for the panel

Is the contribution a feature or a pretraining method? 01 and 06 lead to training-free encodings (walk profiles, Hodge potentials, sketches), which abandons pretraining. 02, 03 and I lead to a learned encoder with a non-trivial objective, judged on label efficiency and cross-ledger transfer. These are different projects with different baselines. The deciding fact is whether any structural signal adds headroom over the Multi-GNN adaptations at full labels. If not, low-label and cross-ledger are the only regimes left, and only a pretraining method can claim them.

## Final verdict

**PIVOT, confidence high. Unchanged.** The panel agrees on the substance; the disagreement is the label, and I hold that replacing the objective is a pivot.

Probability that "exact 12 statistics as node features" matches or beats frozen Fin-PSE (after the pipeline fixes, within seed noise counting as a match): **85%**.

## Ranked ballot of next steps

1. Apply 05's minimum fix list (checkpoint path, PR-AUC score, score all seed edges, keep the encoder in eval mode, one input convention, seeds).
2. Run the pretext audit: held-out per-target R², plus 01's ridge from in-targets to out-targets.
3. Run a small gate on plain and Multi-GIN+EU backbones at 5% and 100% labels: no encoding, exact 12 statistics, random-frozen encoder, Fin-PSE, 4-layer backbone.
4. Adopt 03's relative multi-scale time, quantile amounts, reverse edges and time-respecting sampling.
5. Replace the objective with masked-event and next-window prediction, evaluated by linear probe and leave-one-ledger-out.

## Ranked ballot of alternative directions

1. **Multi-ledger masked and predictive event pretraining** (mine; merges 03-D, 03-C and 02-A). It is the only direction where a learned encoder can beat exact features, and MiNT shows transfer grows with the number of networks.
2. **Leakage-audited benchmark with cross-generator transfer** (05-1; merges 02-B and 05-4). Low risk, and every other direction needs trustworthy numbers first.
3. **Training-free causal-walk encodings** (01-1; merges 03-A and 06-3). Cheap, principled, and the baseline any learned encoder must beat.

**Advise against: amortised motif-count regression** (01-2, 05-2, 06's improvement 4, and my own round-1 alternative 3). The teacher is already fast and exact, so the best outcome is a slower, noisier copy.
