# Rebuttal 01: positional/structural encodings and expressivity

## Where I now agree

- **05 (empirical rigour): the pipeline cannot produce a valid number yet.** I treated the repository as runnable. 05 shows the encodings arm crashes on an empty checkpoint path, PR-AUC ranks by a raw logit, and seed edges into hubs are silently dropped from loss and metrics. My oracle experiment is meaningless until those are fixed, so it moves behind them.
- **05 on `r_2cycle`.** I called it "structurally 2-hop". 05 is right that it is a function of a node's own incident edges; it is 2-hop only for an in-only network. With bidirectional message passing and identity it is a 1-hop quantity, which makes the target list even weaker than I said.
- **03 (temporal graphs): my redesigned targets were mostly static.** Return probabilities diag(P^k) and walk profiles ignore event order. 03's point that B's state is the same for every out-edge of B is an expressivity limit of any node-level encoding, and I did not state it. Targets and inputs need relative time (log Δt to a reference event), not a z-scored absolute timestamp.
- **06 (scalable geometric): cost.** I argued H4 only by naming comparators. 06's observation that the encoder runs inside every batch, so its cost is per step and not precompute-once, is decisive against H4 as implemented. 06's point that return probabilities are near zero on acyclic regions is a real variance problem for my favourite alternative.
- **02 (financial crime): my baseline arm was too weak.** I specified reverse message passing only. The reference row must include ports and ego IDs, and the feature comparator must be mined pattern features, not just the 12 statistics.

## Where I disagree

**1. With 04 (graph SSL), and with 03's improvement 3 and 02's direction A: changing the objective does not fix the containment problem.**
04 says the defect is that the objective is "distillation, not self-supervision", and proposes masked edge-attribute modelling and next-window prediction because those targets are "not a deterministic function of the input". Determinism is the wrong axis. GPSE also distils deterministic targets. What makes an encoding useful is that the target lies outside the function class of the consumer and that the encoder has the means (identity, depth, receptive field) to reach it. An anonymous message-passing encoder trained by masked modelling still outputs a function of the same computation tree the backbone sees, bounded by the same colour refinement. Frozen and concatenated, it is again an optimisation aid, and the controls 04 itself demands (random frozen encoder, deeper backbone) would again erase it. Masked or predictive objectives are worth having only after the encoder is given something the backbone lacks. So I rank target and architecture redesign above objective replacement, and I do not think this is a PIVOT: 04's own direction 3 is the same project done properly.

**2. With 02's direction A (typology-masked motif completion): it is a link-level task posed to a node-level encoder.**
Predicting whether a masked edge closes a temporal cycle requires a representation of the node pair. Node embeddings from an anonymous encoder cannot distinguish pairs whose endpoints are individually indistinguishable; this is the known reason link prediction needs a labelling trick or pairwise features (from memory, not re-verified). Without ego IDs or pair marking, the pretext task will collapse to the degree prior that 02 lists as its risk. With them, the encoder must be re-run per candidate pair, which removes the efficiency argument.

**3. With 06's improvement 5 and experiment arm (e): directed SIGN features are not a neutral test.**
Linear propagation of the exact statistics over {A, Aᵀ, AAᵀ, AᵀA} stays inside what one or two rounds of anonymous bidirectional message passing compute. It cannot count distinct counterparties at two hops or detect a cycle. So if (e) matches the GNN arms, that shows the tested GNNs are not using anything beyond that class, not that "the benchmark is a feature-engineering problem".
## Open question for the panel

Is there any structural signal on AMLworld that helps on top of a backbone with reverse message passing, ports and ego IDs? Egressy et al. prove that class can in principle detect any directed subgraph pattern, so residual gain from an encoding must come from depth, sampling truncation or optimisation. 02 names "nothing helps on top of Multi-GIN+EU" as the dead-end outcome. Nobody on the panel knows the answer, and it decides between redesign and pivot.

## Final verdict

**CONTINUE WITH MAJOR CHANGES, confidence medium** (verdict unchanged; confidence lowered from medium-high). I now attach 05's kill gate: if exact expensive pattern features do not help the adapted backbone, stop.

Probability that the exact 12 statistics as node features match or beat frozen Fin-PSE: **0.85**.

## Ranked ballot of next steps

1. Fix the blockers from 05: checkpoint path, PR-AUC on probability, score every seed edge, encoder kept in eval mode, seeds logged to file.
2. Run the pretext audit (04, mine): held-out per-target R², and out-targets against a ridge from in-targets.
3. Run the gate on Multi-GIN and Multi-PNA with ports and ego IDs (02, 05): plain; exact 12 statistics; mined pattern features; random frozen encoder; depth-matched backbone; Fin-PSE.
4. Only if mined features win: rebuild the encoder (bidirectional, multi-aggregator, resampled random features or ego IDs, relative time per 03, run once on the full graph per 05 and 06) on time-respecting cycle, scatter-gather and pass-through targets.
5. Add a label-scarcity curve and a second generator plus ETH transfer matrix (02, 04) before making any H3 claim.

## Ranked ballot of alternative directions

1. **Time-respecting walk and motif encodings, training-free or sketched** (merging my direction 1, 03's A, 06's 2 and 3). It attacks the actual gap, has theory to build on, is near-linear, and needs no pretraining to be falsified.
2. **Line-digraph message passing over causal transaction pairs** (my 3, 03's B, 06's 4). Three panellists reached it independently, and it is the only proposal that puts event order into the topology; it ranks second because hub blow-up is unresolved.
3. **Hodge / edge-flow encodings** (06's 1). It is the only genuinely positional, global proposal on the table and is cheap, but it depends on the simulator conserving flow realistically.

**Advise against:** in-context relational or tabular foundation models as the main line (04's 2). The evidence is vendor-reported, cyclic patterns fall outside local context sampling, and a result either way teaches nothing about graph structure.
