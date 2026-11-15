# Position paper 06: scalable and geometric graph learning

Scope note. I read all of `src/`, `etl/aml/`, `configs/` and `scripts/`. Nothing was run: the data is absent and torch is not installed here, so statements about PyG sampling semantics rest on my knowledge of the library, not on execution. Web search worked, but arxiv.org was unreachable, so papers below are verified at title/abstract level only unless marked "(unverified from memory)". The cost figures are back-of-envelope estimates, not measurements.

## Verdict

**CONTINUE WITH MAJOR CHANGES** (confidence: high that H4 fails as stated; medium on the overall verdict).

The efficiency claim is not well-posed. The encoder is a learned, sampled, lossy approximation of twelve statistics that the project's own ETL computes exactly in one near-linear scan, so amortisation has negative value. The direction claim is false in the code as written: only in-edges are ever sampled or propagated. What is worth keeping is the question (which structural signals matter on directed temporal multigraphs, and do they transfer), but the answer should be built from exact or sketched precomputed features and from targets that are actually expensive, such as temporal cycles and flow components.

## Strengths

- The target set is multigraph-aware where it counts. `r_2cycle` uses `min(w, w_rev)` over collapsed parallel edges (`etl/aml/transform_window_convert_currencies.py:85-120`), and `deg` versus `fan` separates transaction count from distinct counterparties (`:128-176`). These are the right primitives.
- The ETL is the most scalable artefact in the repository: lazy polars group-bys and one hash self-join, with a CPU-only budget of 16 minutes for all four variants on HI-Medium (`scripts/daic/format_ssl_data.sbatch:5-9`).
- The author's own evidence list already contains the decisive control (item 2: feed the 12 statistics directly).
- Random node inputs to a GNN as a positional-encoding generator is a legitimate, linear-time idea with theory behind it (PEARL, Kanatsoulis et al., ICLR 2025). The project uses it without the parts that make it work (see W4).

## Weaknesses (ranked)

**W1. H4 compares against the wrong baseline, and loses to the right one by orders of magnitude.** GPSE amortises targets that need an eigendecomposition. Fin-PSE amortises group-bys. For HI-Large (about 2.1M accounts, 180M transactions):

| Method | Work (one-off unless stated) | Rough cost |
|---|---|---|
| Exact 12 statistics | hash group-by O(m), sort for gaps O(m log m), hash self-join for reciprocity O(m) | ~10^10 primitive ops, CPU minutes |
| Directed precomputed propagation (A, Aᵀ, AAᵀ, AᵀA; 2 hops; 12 features) | 8 sparse products × m × 12 | ~2×10^10 |
| Sampled RWSE (64 walks, length 16 per node) | n·R·K walk steps | ~2×10^9 steps |
| LapPE or magnetic LapPE, k=16, Lanczos | hundreds of matvecs × m, per split graph | ~10^11 to 10^12, with poor convergence on a graph with many components |
| Fin-PSE, full-graph layer-wise inference | m × 2 layers × ~4×10^3 MACs (the 64×64 `lin_edge` per edge) | ~1.5×10^12 per split graph |
| Fin-PSE as implemented | re-run on every sampled batch, every epoch | ~10^16 |

The last row follows from the code. The encoder is called inside `MPNN.forward` on the sampled batch (`src/networks/mpnn.py:68-75`), with batch size 8192 and fan-out [100, 100] (`configs/aml/aml_large_hi_gin_encodings.yaml:34-35`), and validation and test are both evaluated every epoch (`src/train/aml_train.py:236-237`). That is roughly 22K batches per epoch and 2.2M batches over 100 epochs. Assuming ~10^6 sampled edges per batch, the frozen encoder alone costs ~8×10^9 MACs per batch. So "cheaper than LapPE" is defensible, "cheaper than GPSE" is true but a strawman, and "cheaper than RWSE" and "efficient" are false. H4's cost half fails.

**W2. Direction is not handled natively; half the targets are outside the computational graph.** `NeighborLoader` samples edges pointing into seeds (`src/train/ssl_pretrain.py:23`) and `ResGatedGraphConv` uses the default source-to-target flow (`src/networks/fin_pse.py:33`). I agree with Part B.2. `deg_out`, `fan_out`, `vol_out`, both out-gap statistics and `flow_imbalance` are unobservable at the seed. The only leak is through reciprocated neighbours whose in-sample at hop 2 contains the seed. The L1 loss on these six targets therefore trains the encoder towards conditional medians given in-structure. This damages H1 and H2 directly, and the "handles direction" half of H4.

**W3. The targets that are observable are ones a sum-aggregating sampled GNN approximates badly.** Three separate problems:
- Sampling clips at 100 in-edges, so in-degree and in-volume of hubs are truncated, and hubs are where AML fan-in lives.
- `fan_in` is a count-distinct over a multiset of parallel edges. That is a sketching problem (HyperLogLog), not a sum. The only identity signal is a 1-d Gaussian scalar per node (`src/data/ssl_data.py:78,117`).
- Mean inter-arrival gap telescopes to (t_max − t_min)/(deg − 1), which needs max and min. Gap variance needs a sort. Neither is sum-decomposable in any practical sense.

Part B.1 is right that the targets are 1-hop. I would add that they are 1-hop order statistics and distinct counts, which is the worst case for this architecture and the best case for a group-by.

**W4. The edge encoding carries no information, and the node encoding adds no receptive field.** The encoder never updates edge features: its edge output is `edge_emb(edge_attr)` (`src/networks/fin_pse.py:40,47`), a frozen rank-2 linear map of `[Timestamp, Amount]`, which the backbone already receives (`src/networks/mpnn.py:69,77`). This is not in Part B, and AML is an edge task. Separately, encoder and backbone run on the same 2-hop sampled subgraph, so hop-2 leaves get encodings computed from zero in-edges and hop-1 nodes from one truncated hop. With fixed noise at pretraining and all-ones downstream (Part B.3, confirmed) there is no symmetry breaking at inference, and no pooling over noise draws as PEARL requires.

**W5. Nothing here is geometric, and the baselines are not the directed-multigraph state of the art.** Part B.8 is confirmed. Reverse message passing, ports and ego IDs (Egressy et al., AAAI 2024) are the reference on this benchmark. The Graph Feature Preprocessor (Blanuša et al., arXiv 2402.08593) reports that exact streaming fan, degree, cycle and scatter-gather counts fed to gradient boosting beat GNN baselines on these datasets with higher CPU throughput than a GPU GNN. That is the existing answer to "efficient structural features for transaction graphs", and H1 has to be argued against it.

## Improvements (by value per effort)

1. **Precompute; never encode in the loop.** Join the 12 exact statistics, computed per split graph to respect the chronological split, as node features. An afternoon's work, and it is both the H2 control and the H4 baseline.
2. **Add the reverse direction everywhere.** Append flipped edges with a direction flag (or use separate in/out convolutions) in pretraining, in the encoder and in the backbone. Replace sum-only aggregation with PNA-style sum/max/min/std in the encoder so gap statistics are reachable.
3. **If a learned encoder is kept, run it once with layer-wise full-neighbour inference** and cache the 64-d output per split graph. This removes about four orders of magnitude of compute and the sampling truncation.
4. **Change the targets to things that are expensive exactly.** Candidates: temporal cycle counts of length 3 to 6 within a time window, scatter-gather and biclique counts, 2-hop flow entropy (already computed at `etl/aml/transform_window_convert_currencies.py:208-228` and unused), directed return probabilities, Hodge potential and circular-flow share. Amortisation is only a contribution when the target is superlinear.
5. **Add a decoupled directed baseline.** SIGN-style precomputed features over {A, Aᵀ, AAᵀ, AᵀA} of the exact statistics and of edge-attribute aggregates, then an MLP on endpoint pairs. LightDiC (Li et al., PVLDB 2024) shows the magnetic-operator version reaches papers100M scale.
6. **Either update edges in the encoder or delete the edge-encoding branch.**
7. **Report cost properly.** Wall-clock and peak memory for every row of the table above, on Small, Medium and Large, with sampled-edges-per-batch measured.

## The single experiment that would most change my mind

On HI-Small and HI-Large, GIN and PNA, 5 seeds, PR-AUC and F1, with wall-clock:

- (a) plain backbone;
- (b) backbone plus the 12 exact statistics per endpoint;
- (c) backbone plus Fin-PSE as implemented;
- (d) backbone plus Fin-PSE precomputed full-neighbour with reverse edges;
- (e) directed SIGN features plus MLP, no message passing.

If (c) or (d) beats (b) by more than one standard deviation, the encoder's hidden state holds something the targets do not, and I move to CONTINUE AS PLANNED with fixes 2 and 3. If (b) ≥ (d), which is my prediction, the pretraining contributes nothing and the project should pivot to improvement 4. If (e) matches (a) through (d), the whole benchmark is a feature-engineering problem and the contribution must be the features.

## Alternative research directions

**1. Hodge / edge-flow encodings.** Treat net amount per account pair as an edge flow f. Solve the sparse least-squares problem L₀φ = div f (conjugate gradient, a few hundred sparse matvecs, near-linear). This gives a potential φ per node (upstream/downstream rank) and a gradient-versus-circular split per edge, in windows.
- Why promising: laundering is by construction flow that returns or is conserved through intermediaries. Layering nodes have near-zero divergence and high throughput, and round-tripping is circular flow. Hodge potentials have been used descriptively on real bank transfers (Fujiwara et al., EPJ Data Science 2021) and loop flow on XRP has been linked to laundering (Ikeda and Chakraborty); I found no work using them as learned encodings for AML.
- Risk: the synthetic generator may not conserve flow realistically; on an unfilled clique complex the curl and harmonic parts are not separated.
- First experiment: add φ, per-node divergence and per-edge circular share to (b) above and measure the PR-AUC change on HI-Small.

**2. Exact and sketched temporal-motif features as the encoding, with learning only on top.** Streaming temporal cycle and scatter-gather counts per edge, HyperLogLog sketches for k-hop distinct fan-in/out, count-min for volumes.
- Why promising: linear or near-linear, inductive, streaming-capable, and already competitive (GFP; BlazingAML, arXiv 2604.12241, which I verified by title only).
- Risk: novelty is thin against IBM's line of work; the contribution must be new patterns or guarantees.
- First experiment: sketch-based 2- and 3-hop distinct counts versus exact, measuring error against downstream PR-AUC.

**3. Scalable directed spectral and walk encodings.** Multi-q magnetic Laplacian PEs provably express directed walk profiles (Huang et al., ICLR 2025), but eigendecomposition does not scale. Estimate walk profiles (counts of walks with a given number of forward and backward steps) by Monte Carlo or polynomial filters of the magnetic operator, weighted by amount and constrained to be time-respecting.
- Why promising: a principled directed analogue of RWSE with a sampling estimator and controllable variance.
- Risk: return probabilities are near zero on sparse acyclic regions, so variance per unit of signal is high.
- First experiment: sampled bidirectional walk profiles, R ∈ {16, 64, 256}, as node features in (b).

**4. Line-graph or edge-space message passing with time-respecting adjacency.** Transactions become nodes; (u→v, t₁) connects to (v→w, t₂) only if t₂ > t₁ within a window, so messages follow causal money paths. LineMVGNN (2026) and GRANDE (ICDM 2022) are close neighbours.
- Risk: the line graph has Σ d_in·d_out edges and explodes at hubs; it needs time-window pruning and sampling with an analysed bias.
- First experiment: size of the time-respecting line graph on HI-Small at window sizes of 1 hour to 2 days, before any model.
