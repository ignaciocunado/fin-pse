# Rebuttal 06: scalable and geometric graph learning

## Where I now agree

- **Empirical rigour (05), defects 1 to 4.** Every `_encodings` config crashes on an empty checkpoint path, seed edges into high in-degree nodes are silently dropped from loss and metrics, and PR-AUC ranks by a raw logit. My round-1 cost row "Fin-PSE as implemented, ~10^16" priced a pipeline that does not start. The ordering in my table stands, but that row is an extrapolation, not a description of anything runnable.
- **Financial crime (02), weakness 6a.** Edge ids are concatenated into a float32 tensor, so on Large (180M edges, above 2^24) the seed mask matches wrong edges. This is a scale bug in my own lane and I missed it.
- **PSE theory (01), weakness 1.** "Encoder plus backbone equals a 4-layer in-directed MPNN on a 2-hop sample" is a cleaner statement of my W4, and the 4-layer backbone is the right capacity control. I adopt it.
- **Temporal (03), weakness 2.** My Hodge and decoupled-propagation proposals were static: net flow per window ignores whether money arrived before it left. I also concede a weakness in my own favourite direction. The divergence that drives the Hodge potential is `vol_in − vol_out`, the unnormalised `flow_imbalance` already in the target list, so the potential is a global smoothing of a 1-hop statistic. Only the circular-flow share is new, and it needs time-respecting construction to mean anything.
- **05's correction** that `r_2cycle` is also a 1-hop quantity: agreed, all 12 targets are.

## Where I disagree

**1. PSE theory (01): "RWSE on a sparse graph is K sparse mat-vecs" is wrong, and improvement 3 is dominated by 01's own alternative.** K mat-vecs give one vector's propagation, not diag(P^k). The exact diagonal needs n probes or dense powers. A stochastic estimate needs M probes, so M·K mat-vecs with variance that is worst at hubs. That matters for the H4 comparison: RWSE is cheap only as an estimator, and its cost must be quoted with its error. Separately, 01 proposes a 6 to 8 layer encoder with resampled random features averaged over several draws on the full window graph. That costs M × depth full-graph passes plus pretraining, and what it learns to approximate is what 01's alternative 1 (training-free random probes) estimates directly in O(MK|E|) with a known variance. I would skip the learned version until the probe version is shown to be insufficient.

**2. Graph SSL (04): PIVOT to masked-event, multi-ledger pretraining replaces one unvalidated premise with a larger one.** Masked amount and time buckets from a 2-hop sampled context are again local targets; the likely solution is the marginal distribution conditioned on degree. The multi-ledger corpus does not exist for AML: two or three simulators plus token networks whose semantics differ. And nobody on the panel has yet shown that any structural feature beyond a reverse-message-passing backbone has headroom on this benchmark. I would not spend weeks on a foundation-model objective before that is known. 04's alternative 2 (in-context tabular models) I advise against below.

**3. Empirical rigour (05): the required protocol is unaffordable as specified.** Seven arms, ten seeds, per-arm tuning, four AML datasets. The author's own budget for one Large run is 168 hours (`scripts/daic/aml_large_hi.sbatch:5`). That is hundreds of GPU-weeks on Large alone. The protocol is right for HI-Small and LI-Small. On Large it is feasible only if encodings are precomputed and the heavy arms are decoupled models (precomputed propagation plus MLP), which train in minutes. Rigour and scalability are the same problem here, and the fix list should say so.

**4. Temporal (03): predictive pretraining is not automatically beyond a group-by.** Next-window degree and volume are strongly autocorrelated, so a persistence baseline (this window's exact statistic) may solve most of it. The proposal needs that baseline before it counts as a harder objective.

## Open question for the panel

Is there any headroom at all? Concretely: on a backbone with reverse edges, do exact multi-hop features (temporal cycle and scatter-gather counts, 2-step flow entropy) improve PR-AUC over exact 1-hop statistics? This is 01's oracle test. If the answer is no, every amortisation, motif-pretraining and encoding direction on the table is moot for this benchmark, and the contribution has to be evaluation (05) or transfer (04, 02). If yes, the remaining question is only cost, which is measurable.

## Final verdict

**CONTINUE WITH MAJOR CHANGES**, unchanged, confidence medium. It has moved towards PIVOT: I now think the learned encoder should be dropped unless it beats exact features in the gate experiment, and the surviving project is "which precomputed directed, time-respecting features matter, and what do they cost".

Probability that exact 12 statistics as node features match or beat frozen Fin-PSE: **0.85**. The residual 0.15 is mostly the chance that unfrozen BatchNorm and extra depth give an optimisation benefit that looks like a win.

## Ranked ballot of next steps

1. Repair the harness on Small: score every seed edge, integer edge ids, softmax-based average precision, seeds logged to file (05, 02).
2. Add reverse edges with a direction flag to backbone and encoder (all panellists).
3. Run the three-arm gate on HI-Small and LI-Small: exact 12 statistics precomputed per split graph, frozen Fin-PSE, random frozen encoder, plus 01's 4-layer backbone control.
4. Run the oracle headroom test: exact multi-hop features on the reverse-edge backbone (01).
5. Switch to relative log time and log amounts, and precompute any encoding once per split graph (03, 05, 01, mine).

## Ranked ballot of alternative directions

1. **Training-free, time-respecting walk and random-probe encodings** (01 alternative 1, 03 alternative A, my direction 3, merged). Linear cost, no pretraining, direction and causality built in, and an estimator whose variance can be analysed; it is the honest version of the repository's tagline.
2. **Exact or sketched temporal-motif features with learning on top** (02 E, 01 alternative 2, 04 alternative 3, 05 direction 2, my direction 2, merged). Strongest known signal on this benchmark, but ranked second because the exact miner is already fast, so the amortised variants may have nothing to save.
3. **Time-respecting line-digraph message passing** (01 alternative 3, 03 alternative B, my direction 4). The only proposal that puts causal ordering into the topology; third because hub blow-up makes it a scaling research problem before it is a modelling one. My Hodge circular-flow share belongs here as an edge feature, not as a standalone direction.

**Advise against:** 04's in-context relational or tabular foundation model. A context of a few thousand rows at 0.1% prevalence holds a handful of positives, row-wise context cannot represent cycles, and the supporting evidence is vendor-reported.
