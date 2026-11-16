# Rebuttal 02: financial-crime graph ML

## Where I now agree

- **Empirical rigour (05)** found three things I missed, and each outranks my own evaluation points. Every `_encodings` config has an empty checkpoint path, so the treatment arm has never run. PR-AUC is ranked on the raw class-1 logit (`src/train/aml_train.py:55`), which makes one of the two headline metrics wrong for every arm. And the encoder runs on the truncated mini-batch, so its output depends on the sampler. My round-1 paper treated the repository as closer to runnable than it is.
- **05 and scalable-geometric (06)** are right that all twelve targets are 1-hop, reciprocity included: `r_2cycle` is `min(w, w_rev)` over a node's own edges. I called it the one target that "needs reciprocity information" as if it were structural. It is a join.
- **PSE theory (01)** sharpened my weakness 1 with the right control: encoder plus backbone is a 4-layer in-directed MPNN, so the fair comparison is a 4-layer backbone. I adopt it. 01 and 06 also noted that the edge "encoding" is a frozen rank-2 linear map; on an edge-classification task that is half the edge channel carrying nothing.
- **Temporal graphs (03)** convinced me the timestamp unit mismatch is worse than the amount mismatch I emphasised, and total on ETH.
- **06's cost table** settles H4 better than my one sentence did: the honest comparator is the group-by.
- **What I concede.** My Multi-GNN and GFP scores were from memory and nobody else on the panel verified them either; the author must take them from the papers. My float32 edge-id point (`src/util.py:10-18`) was not confirmed by 05's audit, which only checked the bookkeeping on contiguous train indices. I still believe it applies above 2^24 edges, but it is unexecuted.

## Where I disagree

**1. With 03 and 05: causal (time-respecting) sampling should not become the primary protocol.** Both describe aggregating later edges as look-ahead that is "at odds with" deployment. For card fraud that is right. For AML it is not. Transaction monitoring is retrospective: alerts are raised in batch over look-back windows of days to weeks, and investigators file within a regulatory period measured in weeks. The defining evidence of layering is what happens *after* the suspicious credit, namely the onward transfer. A sampler restricted to edges before t removes the outflow half of every fan-in-then-fan-out and every cycle closure for the earliest edges in the pattern. Report both protocols, but the retrospective one is the operationally relevant one, and it is the only one comparable to published numbers.

**2. With graph-SSL (04): PIVOT is premature, and masked edge-attribute modelling is the wrong replacement on this benchmark.** The gate experiment costs about a week after 05's fixes; pivoting before running it discards the cheapest information available. On the replacement: in AMLworld, amount, time gap and payment format are draws from generator priors. Masked-attribute prediction will learn those marginals well and transfer them nowhere. The laundering signal in these datasets is topological and order-dependent, and payment format is a known near-shortcut for the label (unverified from memory), so a format-prediction head risks pretraining the shortcut. 04's own direction 3 (amortised motif counts) is the better objective and I would swap its ranking with direction 1.

**3. With 04's in-context tabular foundation model direction.** At roughly 0.1% illicit edges (0.05% on LI), a context of a few thousand rows holds two or three positives. It only works with heavy stratified context construction, which is a supervised pipeline under another name, and the supporting result is vendor-reported on a different benchmark.

**4. With 06's Hodge direction as stated.** "Layering nodes have near-zero divergence and high throughput" also describes every salaried current account and every payment processor. Without time-respecting, short-window flows the feature separates businesses from consumers, not launderers from businesses. AMLworld is also multi-currency and does not conserve value across hops. Worth one afternoon as extra columns in the oracle test; not a research direction.

**5. With 05's claim that published numbers "are likely not comparable".** I checked the reference implementation: it drops unsampled seed edges for these datasets in the same way. Published numbers share the bias, so they are comparable with each other and wrong in the same direction. That weakens the case for 05's benchmark-first direction as the main contribution, though the audit is worth an appendix. 05's reviewer protocol (seven arms, ten seeds, per-arm tuning, four datasets including Large) is also beyond a single author's compute; five seeds on Small and Medium is the standard this literature accepts.

## Open question for the panel

How much headroom on AMLworld is structural at all? A large share of illicit edges belong to no injected pattern. If the residual errors of Multi-PNA+EU or GFP+XGBoost are concentrated in those unpatterned edges, no structural encoding, learned or exact, can help, and every "redesign the targets" proposal on this panel (mine included) is aimed at solved cases. A per-typology recall breakdown of a strong existing model against the pattern files answers this in two days and should precede any redesign.

## Final verdict

**CONTINUE WITH MAJOR CHANGES**, confidence medium-high. Unchanged, but now explicitly gated as 05 proposes: if the control experiment fails, pivot.

Probability that "exact 12 statistics as node features" matches or beats frozen Fin-PSE: **0.85**.

## Ranked ballot of next steps

1. Apply 05's blocking fixes: checkpoint path, PR-AUC score, direct seed-edge scoring, encoder kept in eval mode, one input convention across phases.
2. Per-typology recall of a reverse-MP backbone on HI-Small and LI-Small against the pattern files (the open question above).
3. Gate experiment, 5 seeds, reverse-MP backbone: plain; 4-layer (01); exact 12 statistics; random frozen encoder; Fin-PSE; plus a GFP+XGBoost row.
4. Per-target held-out R² for the encoder, with the in-targets ridge baseline for out-targets (01, 04).
5. Oracle test: exact multi-hop typology counts and 2-step flow entropy as features (01). Only if this beats the gate arms, redesign the encoder (bidirectional, cached full-graph inference, relative time) and add a label-scarcity curve.

## Ranked ballot of alternative directions

1. **Amortised temporal-motif / typology encoder** (01 #2, 04 #3, 05 #2, 06 #2, my A). It keeps the author's pretrain-and-reuse framing, targets what laundering looks like, and has a strong exact baseline that makes the result falsifiable.
2. **Cross-generator and cross-institution transfer with invariant inputs** (my B, 05 #4, 04 #4, 03 D). It is the question practitioners actually have, and the only setting where a learned encoder has a plausible edge over feature engineering; it ranks second because synthetic-to-synthetic results may persuade nobody.
3. **Time-respecting edge-space message passing** (01 #3, 03 B, 06 #4, my C). It builds flow causality into the topology, which no node encoding can, but hub blow-up makes it the riskiest engineering of the three.

**Advise against:** 04's in-context relational/tabular foundation model direction, for the prevalence reason given above.
