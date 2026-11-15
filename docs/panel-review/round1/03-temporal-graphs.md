# Position paper 03: temporal and dynamic graph learning

Scope: I read all of `src/`, `etl/aml/*.py`, the ETH and AML notebooks and the configs. Nothing was executed (PyTorch is not installed in this environment and the data is absent), so every code claim below comes from reading the source and from PyG's documented semantics. Web search was available for part of the session; citations I could not check are marked.

## Verdict

**CONTINUE WITH MAJOR CHANGES** (confidence: high that the temporal half of H4 is false as implemented; medium on the overall verdict).

The encoder does not handle time natively. It treats one z-scored absolute timestamp as an ordinary edge scalar on a static multigraph, with a time-agnostic sampler, and the unit of that scalar changes by a factor of several (AML) to roughly a hundred or more (ETH) between pretraining and downstream use. The idea of a cheap, reusable, direction- and time-aware node encoding is still worth pursuing, but only if time becomes relative, multi-scale and causal, and if the pretraining task stops being a function the input already contains.

## Strengths

- The data model is right: a directed multigraph with one edge per transaction and no snapshot collapsing, so event-level time is still available to a better encoder.
- Downstream splits are chronological, and the train, validation and test graphs are nested prefixes (`src/data/aml_data.py:229-251`, `src/data/eth_data.py:43-45`). Normalisation statistics come from the training split only (`aml_data.py:253-256`, `eth_data.py:61-63`). Training therefore never sees validation or test edges.
- ETH evaluation is inductive in the strict sense: test nodes first appear after the 80th percentile (`eth_data.py:39-41`). Few financial-graph papers do this.
- Pretraining uses no labels and a different simulator run (HI-Medium) from the downstream sets, so there is no label leakage from pretraining. The windowing question is therefore not about leakage at all (see weakness 5).
- Inter-arrival statistics are a sensible first temporal target family: burstiness is a real signal for both laundering and phishing.

## Weaknesses (ranked)

**1. Time is an absolute, linearly embedded scalar whose unit and support change between pretraining and deployment.** (Breaks H3 and the time half of H4; weakens H1.)
- Pretraining z-scores the timestamp per 2-day window (`src/data/ssl_data.py:122-124`), so the encoder only sees position within a window. One unit is about 0.58 days if events are uniform.
- AML downstream z-scores with train-split statistics over the whole training period (`aml_data.py:253`). ETH does the same over a training span of years; the dataset covers 1,261 days (`etl/eth/eda.ipynb`). The frozen encoder receives column 0 unchanged (`src/networks/mpnn.py:69`). My estimate is a unit mismatch of a few times on AML and two orders of magnitude on ETH. Whatever the encoder learned about gaps is miscalibrated by that factor.
- Under a chronological split every validation and test timestamp lies above the training maximum. On ETH this is total: a test node's incident edges all have `timestamp > t2 > t1`, outside anything the backbone or encoder was fitted on. A `nn.Linear` on raw time (`src/networks/fin_pse.py:27`) extrapolates linearly there.
- Part B item 4 is correct and understates the problem: the amount column shifts too, but the time column shifts by construction.

**2. No causal, time-respecting structure is represented.** (Breaks the "money flow" motivation; weakens H2.)
- For A→B at t1 and B→C at t2, the path carries money only if t1 < t2. The encoder computes one static state per node. B's layer-1 summary of its in-edges is the same for every out-edge of B (`fin_pse.py:42-44`), so it cannot condition on "before t2".
- None of the 12 targets depends on the order of events across two nodes. `r_2cycle` counts reciprocal pairs within a window regardless of order (`etl/aml/transform_window_convert_currencies.py:85-120`). Temporal motifs (Paranjape et al., 2017), causal anonymous walks (Wang et al., 2021) and temporal walk matrices (TPNet, Lu et al., NeurIPS 2024) are the established tools for this, and none is used.

**3. The samplers are time-agnostic, and half of each node's history is invisible.** (Breaks H2 for the out-direction targets; biases H1 evaluation.)
- `NeighborLoader` and `LinkNeighborLoader` are called without `time_attr` (`src/train/ssl_pretrain.py:23`, `src/train/aml_train.py:23-43`, `src/train/eth_train.py:23-45`). A validation or test edge at time t aggregates edges later than t. This matches the Multi-GNN convention, but it is retrospective scoring, not detection at transaction time, and the paper must say so.
- I confirm Part B item 2 by reading: the loaders sample in-edges of seeds and `ResGatedGraphConv` passes source to target, so out-degree, fan-out, out-volume and outgoing gaps are outside the seed's computational graph except through reciprocal edges. Four of the six temporal and directional target pairs are fitted from noise.
- Uniform sampling of 100 in-edges breaks gap statistics for hubs: a subsample's mean gap is inflated by roughly deg/100.
- A side effect in AML evaluation: predictions are kept only for seed edges that happened to be sampled (`aml_train.py:104, 162`). Edges into nodes with in-degree above 100 are dropped at random, so the test set is a degree-biased subsample.

**4. The temporal targets are poorly posed.** (Weakens H2.)
- Mean gap is `(t_last - t_first)/(n-1)`, a function of degree and span. Gap variance needs order statistics; one linear time input and one ReLU-gated sum (`fin_pse.py:33`, the gate activation is overridden to ReLU) gives at best a coarse piecewise-quadratic time histogram.
- Variance is in seconds squared, untransformed, then z-scored (`ssl_data.py:127-131`), so the L1 loss fits the median and ignores the tail that matters.
- Nodes with fewer than two events get −1 mean, and nodes with exactly two events get −1 variance (`transform_window_convert_currencies.py:79`), mixing "undefined" into a continuous scale. Part B item 7 is correct.
- Timestamps have one-minute resolution, so many gaps are exactly zero.
- Tumbling windows cut gaps at boundaries and discard recency (time since last event), which is usually the most predictive temporal feature.

**5. Windowing is justified by the wrong argument.** (Weakens H2 ablations.)
- The brief says windows limit time-based leakage. The pretraining task is not predictive: targets are computed from the same window's edges that form the input. There is nothing to leak, and nothing temporal to learn beyond aggregation.
- `sixty` and `eighty` cut-offs are computed and never used (`transform_window_convert_currencies.py:20-22`).
- The cumulative variant changes the timestamp scale in every snapshot because normalisation is per snapshot.

**6. Non-stationarity is unmeasured.** (Weakens H1 and H3.)
- AMLworld has a long sparse tail: in HI-Medium 80% of rows fall before 14 September but the data runs to 28 September (`etl/aml/eda.ipynb`). I recall that the tail is laundering-heavy (unverified from memory); if so, absolute time is a label shortcut and the frozen encoder passes it through.
- ETH train nodes are observed up to `t1`, test nodes up to the end of the data, so observation length differs systematically between splits.

## Improvements (by value per effort)

1. **Replace absolute time with relative, multi-scale time.** Encode `Δt = t_ref − t_e` in seconds, where `t_ref` is the seed edge's time (AML) or the node's last event (ETH and pretraining). Use `log1p(Δt)` plus a fixed cosine encoding with log-spaced frequencies (GraphMixer style, no learned parameters). No per-window or per-split z-scoring. About 30 lines; it removes weaknesses 1 and most of 6.
2. **Make sampling time-respecting and bidirectional.** Pass `time_attr` and `edge_label_time` to the loaders with `temporal_strategy="last"`, and add reverse edges with a direction flag so out-history is visible. Report both the retrospective and the causal protocol.
3. **Make pretraining predictive.** Input: edges in `[t−W, t)`. Targets: statistics of `[t, t+H)` (next-gap, future degree and volume, whether a reciprocal or return transfer arrives within δ). Add masked edge-time and amount reconstruction. This is the first version of the objective that cannot be solved by a group-by.
4. **Add order-dependent targets.** Count of time-respecting 2- and 3-hop out-paths within δ, δ-temporal 2- and 3-cycles, fan-in followed by fan-out within δ, pass-through ratio (out-volume within δ of in-volume), and recency. Log-transform heavy tails; predict a validity mask separately instead of filling −1.
5. **Report metrics by day and by time since training cut-off**, and fix the dropped seed edges by forcing seed edges into the batch.
6. **Compare against a stream baseline.** DyGFormer or GraphMixer on first-hop history with the same features, and a Graph Feature Preprocessor style temporal-pattern baseline with gradient boosting (unverified from memory that this is the strongest reported tabular baseline on AMLworld).

## The single experiment that would most change my mind

A time-shuffle control. Train the full pipeline (pretraining and downstream) four ways: (a) as implemented; (b) timestamps permuted among each node's incident edges, preserving the marginal; (c) the timestamp column removed; (d) improvement 1 applied. Five seeds, PR-AUC broken down by test day.

- If (a) ≈ (b) ≈ (c) and (d) is clearly better, the current encoder carries no usable temporal information and the "native time" claim is dead until the encoding changes. This is what I expect.
- If (a) beats (b) and (c) by more than seed noise on late test days and on ETH, I am wrong about weakness 1 and would move to CONTINUE AS PLANNED on the temporal side.
- If (c) beats (a), absolute time is a harmful shortcut and the redesign is urgent.

## Alternative research directions

**A. Causal-walk structural encodings for transaction graphs.**
- Idea: node-pair and node encodings from time-respecting walk counts with time decay, maintained incrementally by random projection (the TPNet construction), extended to directed multigraphs with amounts as walk weights.
- Promise: this is the temporal analogue of RWSE, is cheap, and directly encodes layering and round-tripping.
- Risk: hub-dominated counts; simulator patterns may be too easy.
- First experiment: decayed temporal-walk return probabilities for k ≤ 4 as node features into the existing GIN, against RWSE on the static graph.

**B. Higher-order (line-graph or De Bruijn) message passing over causal edge pairs.**
- Idea: nodes of the lifted graph are transactions; connect e1=(A→B, t1) to e2=(B→C, t2) only if 0 < t2−t1 ≤ δ. Run a GNN there. Edge classification becomes node classification and every message follows a feasible money path.
- Promise: causality is built into the topology, not learned.
- Risk: lifted-graph size at hubs, which needs δ and top-k pruning.
- First experiment: δ ∈ {1h, 1d}, 2-layer GIN on the lifted HI-Small graph versus the current baseline.

**C. Event-sequence transformer per account with neighbour co-occurrence features.**
- Idea: no message passing. Tokenise each endpoint's last L transactions with relative time, amount, direction and counterparty co-occurrence counts (DyGFormer style), pretrain by masked-event modelling.
- Promise: time-respecting by construction, inductive, and pretraining transfers because tokens are relative.
- Risk: misses patterns longer than two hops.
- First experiment: L=64 on HI-Small with the same split; compare to GIN with reverse message passing.

**D. Multi-network temporal pretraining with a proper transfer benchmark.**
- Idea: pretrain across many transaction networks, as MiNT did with 84 Ethereum token networks, but at event level with relative time, and test zero-shot on unseen networks.
- Promise: this answers H3 with real diversity instead of one simulator and one crawl.
- Risk: negative transfer, which DyGFM (2026, abstract only read) reports as common.
- First experiment: pretrain on 20 token networks, test on 5 held out, against training from scratch.

## Sources

- [MiNT: Multi-Network Training for Transfer Learning on Temporal Graphs](https://arxiv.org/abs/2406.10426v3)
- [Transfer Learning for Temporal Link Prediction](https://arxiv.org/html/2504.10925)
- [DyGFM summary: Decoupled and Divergence-Conditioned Prompt for Multi-domain Dynamic Graph Foundation Models](https://www.themoonlight.io/review/decoupled-and-divergence-conditioned-prompt-for-multi-domain-dynamic-graph-foundation-models)
- [Improving Temporal Link Prediction via Temporal Walk Matrix Projection (TPNet)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ff7bf6014f7826da531aa50f4538ee19-Abstract.html)
- [Supra-Laplacian encoding for dynamic-graph transformers (SLATE)](https://arxiv.org/pdf/2409.17986)
- [Provably Powerful Graph Neural Networks for Directed Multigraphs](https://arxiv.org/html/2306.11586v3)
- [GRANDE: a neural model over directed multigraphs with application to anti-money laundering](https://arxiv.org/pdf/2302.02101)
- [Graph Learning on Financial Tabular Data (learnable positional encodings on IBM AML data)](https://repository.tudelft.nl/record/uuid:9e6a55e6-ab50-431d-bd52-b6ffb27b4a70)

Cited from memory, not re-checked this session: Paranjape et al. 2017 (temporal motifs), Wang et al. 2021 (causal anonymous walks), GraphMixer, DyGFormer, the Graph Feature Preprocessor.
