# Position paper 01: positional/structural encodings and expressivity

Literature note: web search worked for part of the session and was then cut off; arXiv could not be fetched directly. Citations marked "verified" were confirmed by search at abstract level. Those marked "from memory" were not.

## Verdict

**CONTINUE WITH MAJOR CHANGES. Confidence: medium-high.**

As implemented, Fin-PSE is not a positional or structural encoding. Downstream, the frozen encoder computes a function that lies inside the backbone's own function class, on the same sampled subgraph, so it cannot add expressive power; and half of its pretraining targets are not functions of its input. The underlying question (can a learned encoder amortise directed, temporal, multigraph structure that a plain MPNN cannot see?) is still open and worth one redesigned attempt.

## Strengths

- The gap is real. LapPE, RWSE and GPSE are undirected and static. Directed PE theory (Multi-q Magnetic Laplacian and walk profiles, Huang, Wang and Li, ICLR 2025, verified) has not been taken to temporal multigraphs with millions of edges.
- The author's own evidence list already contains the decisive control (item 2, targets fed directly as features).
- The ETL is the most valuable asset. It already computes a genuinely 2-hop quantity, the two-step flow entropy (`etl/aml/transform_window_convert_currencies.py:208-226`), which is never used as a target (`src/data/ssl_data.py:36-49`).
- The windowed graph construction is the right unit for temporal targets.

## Weaknesses (ranked)

**1. The encoder is contained in the backbone's function class. Hits H1 and H2.**
Downstream node inputs are all-ones (`src/data/aml_data.py:145`, `src/data/eth_data.py:51`), and no config sets `encodings_random_feats`. With constant inputs, a 2-layer source-to-target sum-aggregation network over `[t, amount]` is an anonymous depth-2 in-tree function. GINE at the same depth, direction and edge features, on the same mini-batch (`src/networks/mpnn.py:68-75`), can already represent it. Encoder plus backbone equals a 4-layer in-directed MPNN restricted to a 2-hop sample. A PSE earns its place by injecting what the backbone cannot compute (global spectrum, closed-walk counts, identity); this one injects nothing of that kind. Any gain is an optimisation effect, and the fair control is a 4-layer backbone, not the 2-layer one.
The edge "encoding" is worse: it is the frozen linear map of two raw features (`src/networks/fin_pse.py:40,47`), rank 2, with no structure in it.

**2. Six of twelve targets are unobservable. Hits H2 and H3.**
Part B item 2 is correct, and it is architectural rather than a sampling artefact. `NeighborLoader` samples in-edges and `ResGatedGraphConv` propagates source to target, so a node's out-edges enter its computation tree only via a directed cycle of length at most 2. `deg_out`, `fan_out`, `vol_out`, `mu_gap_out_sec`, `var_gap_out_sec` and also `flow_imbalance` (six, not the five Part B lists) can only be fitted through their correlation with in-side statistics. L1 then learns a conditional median of the simulator's in/out coupling, which is exactly what will not transfer to Ethereum.

**3. The observable targets are either trivial or mismatched to the aggregator. Hits H2.**
- `deg_in` and `vol_in` are single-layer sums; GINE gets them for free.
- `mu_gap_in` is (t_max − t_min)/(n − 1). It needs max and min, which a sum aggregator only approximates, while the downstream PNA backbone has min, max and std natively (`src/networks/mpnn.py:111`). The encoder is weaker than the model it is supposed to help.
- `var_gap_in` needs order statistics of the timestamp multiset.
- `fan_in` and `r_2cycle` need node identity. Pretraining supplies one fixed scalar per node (`src/data/ssl_data.py:78,117`) through a linear embedding, so all 64 channels are affine in one number. Membership testing ("is my ID in my in-neighbour's in-multiset?") through sums needs many independent random channels (random-feature results of Sato et al. and Abboud et al.; sum-decomposition width bound of Wagstaff et al.; all from memory). Downstream there is no identity at all, so whatever was learned for `r_2cycle` is not computable at inference.

**4. Loss and normalisation erase the tails that carry the signal. Hits H1 and H3.**
L1 on z-scored heavy-tailed counts and volumes is near-constant median regression. Normalisation is per window (`src/data/ssl_data.py:131`), so the same embedding means different absolute behaviour in different windows. The `-1` sentinel for undefined gaps (`etl/aml/transform_window_convert_currencies.py:79`) sits next to true zero gaps at minute resolution. Targets are computed on the full window but inputs are capped at 100 in-edges per hop (`configs/ssl/pretrain_2dwindow_convert.yaml`), so hub targets are censored.

**5. Implementation defects that confound any result.** All verified.
- Part B item 5: `model.train()` at `src/train/aml_train.py:92` re-enables BatchNorm in the "frozen" encoder.
- Part B item 4: per-window USD z-scores in pretraining versus global native-currency z-scores downstream (`src/data/aml_data.py:153,253`).
- Part B item 6: encoder depth is read from `cfg.gnn.layers_mp` (`src/networks/fin_pse.py:32`).
- New: the conv is built with `act=act_dict[cfg.gnn.act]()` (`src/networks/fin_pse.py:33`), so the gate is ReLU, not sigmoid. Messages are an unbounded product of two linear maps of heavy-tailed edge features.

**6. H4 is asserted against the wrong comparators.**
RWSE on a sparse graph is K sparse mat-vecs. The real cost competitor is exact pattern mining: the Graph Feature Preprocessor (arXiv 2402.08593, verified) reports up to 36% minority-F1 gain over GNN baselines with gradient boosting on mined cycle and scatter-gather features. Multi-GNN (Egressy et al., AAAI 2024, verified) gets directed-pattern power with reverse message passing, ports and ego IDs and no pretraining. Among learned encoders, GFSE (arXiv 2504.10917, verified) and PEARL (Kanatsoulis et al., ICLR 2025, verified) are the comparators; a 2025 thesis already applies PEARL to the IBM AML data (surfaced by search, not read).

## Improvements (by value per effort)

1. **Probe before anything else (one day).** Report per-target R² on a held-out window, and R² of out-targets against a ridge regression from in-targets. I predict the encoder does not beat the ridge on the six out-targets.
2. **Add the controls.** Exact targets as features; an untrained encoder; a 4-layer backbone; the backbone with reverse message passing.
3. **Fix the encoder.**
   - Bidirectional message passing with separate in and out weights.
   - PNA-style aggregators (sum, min, max, std) or a time-sorted sequence aggregator.
   - Random features of 16 to 32 dimensions, resampled every forward pass in pretraining and downstream, averaged over several draws (the PEARL recipe).
   - Depth 6 to 8, decoupled from the backbone.
   - Encodings computed once per window on the full window graph and cached, not on the backbone's mini-batch.
   - Drop the edge output, or define it as a function of endpoint encodings and relative time.
4. **Replace the targets with quantities outside the backbone's class.**
   - Directed return probabilities diag(P^k) and diag((Pᵀ)^k), k = 2..8.
   - Walk-profile counts over forward/backward step patterns up to length 4.
   - Time-respecting cycle counts of length 2 to 6 within a window Δ.
   - Scatter-gather, fan-in-then-fan-out and biclique counts.
   - Forward and reverse personalised-PageRank mass.
   - The existing `H_out_2_amt`, `supp_out_2_amt`, `pmax_out_2_amt`.
   - Keep degree and volume only as auxiliary heads. Use `log1p`, Huber plus cosine loss (as GPSE did, from memory), global normalisation, and a mask instead of a sentinel.
5. **Fix the defects in weakness 5**, and encode time as within-window deltas and amount as log or quantile so the input scale is shared across datasets.

## The single experiment that would most change my mind

An oracle-PSE test on HI-Small and LI-Small, 5 seeds, PR-AUC and F1.

- (a) GIN and PNA with reverse message passing.
- (b) (a) plus the exact redesigned targets from improvement 4 as node features.
- (c) (a) plus the current frozen Fin-PSE.
- (d) (a) with 4 layers and no encoder.

If (b) does not beat (a) and (d) by a clear margin (more than about 2 F1 points, outside seed variance), no learned encoder of these targets can help and the PSE premise is dead on this benchmark: I move to PIVOT. If (b) wins clearly, the project reduces to whether a learned encoder recovers most of that gap at lower cost on Large and on Ethereum, and I move toward CONTINUE. If (c) beats (d), my weakness 1 is wrong in practice and I would want to know why.

## Alternative research directions

1. **Training-free random-probe encodings for directed temporal multigraphs.**
   - Idea: propagate M random probe vectors along forward/backward step patterns with time-respecting masks and pool second moments. This gives unbiased estimates of walk-profile entries and return probabilities in O(MK|E|), with no pretraining.
   - Promise: joins PEARL's sample pooling to walk-profile theory, and scales where eigen-decomposition does not.
   - Risk: estimator variance at hubs.
   - First experiment: the synthetic directed-pattern tasks of Egressy et al., then HI-Small, against RWSE and Multi-GNN.

2. **Amortised temporal-motif counting.**
   - Idea: train a network to regress exact mined pattern counts (temporal cycles, scatter-gather), with error bounds as a function of depth and identity scheme, as a streaming substitute for the miner.
   - Promise: turns H4 into a measurable accuracy-versus-throughput curve against a strong, deployed baseline.
   - Risk: the exact miner is already fast on CPU.
   - First experiment: count regression on HI-Medium windows, then downstream F1 with predicted versus exact counts.

3. **Transaction-centric line-digraph learning.**
   - Idea: transactions become nodes; arcs join (u→v, t1) to (v→w, t2) when t2 > t1 within Δ. Edge classification becomes node classification, time-respecting flow becomes plain adjacency, and standard directed PEs apply.
   - Promise: the temporal ordering constraint, which no node-level PSE expresses, is built into the topology.
   - Risk: quadratic blow-up at hubs.
   - First experiment: capped-fan line digraph on HI-Small with GIN and directed RWSE.

4. **Expressivity map of laundering typologies.**
   - Idea: for each AMLworld typology, prove the minimal depth, direction and identity mechanism needed for detection, and release a synthetic suite.
   - Promise: gives every encoding paper on this benchmark a principled target list.
   - Risk: a theory contribution with modest empirical headline.
   - First experiment: per-typology recall of plain, reverse-MP and ego-ID models.
