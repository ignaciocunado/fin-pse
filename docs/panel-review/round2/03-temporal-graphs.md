# Round 2 rebuttal 03: temporal and dynamic graph learning

## Where I now agree

- **01 (PSE theory), weakness 1.** The frozen encoder sits inside the backbone's own function class on the same sampled subgraph. My round-1 paper treated the time encoding as the main defect; the containment argument is more basic. Fixing time alone would still leave an encoder that adds nothing the backbone cannot compute.
- **05 (empirical rigour).** The encodings arm cannot start (`encodings_file: ''`), PR-AUC is computed on a raw logit, and the encoder's BatchNorm is not frozen. I missed all three. My time-shuffle experiment is meaningless until they are fixed, and I now accept 05's kill gate.
- **06 (scalable).** The encoder is re-run on every batch, so the cost claim in H4 fails on arithmetic alone. 06's first experiment for the line-graph direction (measure the size of the time-respecting line graph before building a model) is better than the one I proposed.
- **04 (graph SSL).** "Log inter-event time to the endpoint's previous event" is a cleaner statement of the relative-time input than mine, and the label-scarcity curve is the regime where pretraining could matter.
- **Concessions.** I wrote that four target pairs are unobservable; 01, 05 and 06 are right that it is six of twelve targets, because `flow_imbalance` depends on out-volume. My "single experiment" is a useful arm, not the decisive test. The decisive test is the exact-statistics control that 02, 04, 05 and 06 all name.

## Where I disagree

**1. With 01, 05 and 06: "precompute the encoding once per split graph and cache it" is wrong for a temporal edge task.**
A cached vector per node over the whole test graph means a transaction on test day 1 is scored with statistics that include test day 5. That turns the benchmark's conventional look-ahead into a stored feature. It also gives every transaction of an account the same vector, which removes exactly the per-event conditioning that layering detection needs. 06's qualifier, "per split graph to respect the chronological split", protects the split boundary but not ordering inside the split. Encodings and the exact-statistics control must be computed as of the scored transaction's time, over a trailing window.

**2. With 01, 02 and 05: the windowing is not "the right unit" and does not show sound thinking about leakage.**
The pretext targets are computed from the same window that forms the input, so there is no leakage for windows to limit. Tumbling 2-day windows anchored to the epoch cut inter-arrival gaps at arbitrary boundaries and drop recency. Per-window z-scoring changes the time unit between pretraining and deployment. 01's fix, "within-window deltas", keeps the arbitrary boundary. Time should be relative to the scored event, not to a window edge.

**3. With 04: the PIVOT to masked edge-attribute modelling is overstated.**
Masking an edge's time bucket is close to trivial when the neighbouring edges of the same account carry their own times. Masked amount and payment format on AMLworld mostly teach the simulator's marginals, which is the artefact 04 itself warns about. 04's second head, predicting next-window behaviour from the current window, is the part worth keeping because it is the only proposed objective that is not a function of the input. I also do not think the verdict should be PIVOT: every replacement 04 proposes reuses this pipeline.

**4. With 06: Hodge encodings on net flow per account pair discard the signal.**
Netting A→B and B→A within a window cancels round trips, and any aggregation over a window loses whether money left before or after it arrived. Circular flow on a time-collapsed graph includes cycles that no money could have followed. The idea needs a time-respecting formulation before it is a temporal encoding.

**5. With 02: switching to the adapted backbone first is the wrong order.**
Reverse message passing and ego IDs should be an arm, but the gate should run on the plain backbone too. Otherwise a null result cannot be separated from "the adaptations already cover it".

## Open question for the panel

Over what time horizon are the "exact 12 statistics" computed for the control? Whole split graph, trailing 2-day window as of each transaction, or all history up to the transaction? The three give different features and different amounts of look-ahead, and the pretraining used the 2-day version. The panel has agreed on this control as decisive without agreeing on what it is. The same choice decides whether the benchmark is reported in the conventional retrospective protocol, a causal one, or both.

## Final verdict

**CONTINUE WITH MAJOR CHANGES**, confidence medium. Unchanged in label, but I now attach 05's kill gate: if frozen Fin-PSE does not beat the exact-statistics and random-encoder arms, the pretrained-encoder framing should be dropped in favour of the directions below.

Probability that the "exact 12 statistics as node features" control matches or beats frozen Fin-PSE: **0.85**.

## Ranked ballot of next steps

1. Fix the blockers from 05: checkpoint path, PR-AUC score, direct scoring of all seed edges, encoder kept in eval mode, five or more seeds.
2. Run the pretext audit from 01 and 04: held-out per-target R², and out-targets against a ridge from in-targets.
3. Run the gate: plain backbone, exact statistics as of transaction time, random frozen encoder, Fin-PSE, with a timestamp-shuffled arm, on plain and reverse-message-passing backbones.
4. Make inputs unit-free and causal: relative log time, log amount, reverse edges with a direction flag, time-respecting sampling (mine, 04, 05).
5. Replace the targets with order-dependent, expensive ones and add a next-window predictive head (01, 02, 04, mine).

## Ranked ballot of alternative directions

1. **Time-respecting motif and causal-walk encodings, exact, sketched or amortised** (01 directions 1 and 2, 04 direction 3, 05 direction 2, 06 directions 2 and 3, my A). It ranks first because it is the only direction that keeps the encoder idea while giving it targets outside the backbone's reach and outside a group-by.
2. **Time-respecting line-graph message passing** (my B, 01 direction 3, 06 direction 4, with 02's flow-conservation idea C as a variant). It ranks second because causality is built into the topology, but hub blow-up is unmeasured and could end it early.
3. **Leakage-audited benchmark with a causal protocol and cross-generator transfer** (05 directions 1 and 3, 02 direction B, 04 direction 4). It ranks third because it is low risk and every other direction needs it, but it is not a method contribution.

**Advise against:** 04 direction 2, in-context relational foundation models as the research direction. The supporting evidence is vendor-reported, local context sampling cannot see multi-hop temporal cycles, and it leaves the author with nothing of their own. It is fine as one baseline row.
