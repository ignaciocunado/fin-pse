# Position paper 02: financial-crime graph ML

Note on sources: web search worked for part of this review; arXiv itself was unreachable from the session, so I could not open the primary tables. Numbers below are marked with where they came from.

## Verdict

**CONTINUE WITH MAJOR CHANGES** (confidence: medium-high).

The question "can a pretrained, reusable structural encoder replace per-dataset feature engineering for laundering detection" is worth asking, and I found no published work that answers it on AMLworld. But as built, the encoder cannot see more than the backbone it feeds, its targets are a subset of features already published as a baseline for this benchmark, and the comparison is against a backbone the field stopped using three years ago. H1 as stated is nearly unfalsifiable in an interesting way: either result tells us little.

## Strengths

- **The right benchmark and protocol skeleton.** The loader, the day-level 60/20/20 temporal split and the seed-edge evaluation follow the Multi-GNN reference implementation (`src/data/aml_data.py:167-201`, `src/train/aml_train.py:100-104`), so results can in principle sit next to published ones.
- **Edge-level AML plus node-level ETH** is the standard pairing in this literature, and ETH is a real graph, which the project needs badly.
- **The author's own evidence list is correct.** Item 2 (feed the targets directly) is the decisive control and the author named it unprompted.
- **Windowed pretraining** (`etl/aml/transform_window_convert_currencies.py:27-35`) is the right instinct: banks score on rolling windows, and laundering in AMLworld is temporally local.
- **Label-free pretraining matches practice.** Banks have billions of unlabeled transactions and a few thousand confirmed SARs. A project that showed gains in the low-label regime would matter.

## Weaknesses (ranked)

**1. The encoder is informationally dominated by the backbone.** I verified Part B.2. Pretraining uses default `NeighborLoader` (`src/train/ssl_pretrain.py:23`), which samples in-edges, and `ResGatedGraphConv` (`src/networks/fin_pse.py:33`) passes source to target. Half the targets (out-degree, fan-out, out-volume, both outgoing gap statistics) are not in a seed's computation graph except by accident. Downstream the encoder runs on the same sampled subgraph as the backbone, with the same depth (`fin_pse.py:32`), the same direction, and fewer edge features (`src/networks/mpnn.py:69`). Every function the frozen encoder computes, a 2-layer GINE could compute from its own inputs. In-degree in particular is exactly what one sum-aggregation over all-ones features (`aml_data.py:145`) yields. So any H1 gain is an optimisation effect, not new information. Kills H2; makes H1 weak even if positive.

**2. The targets are not what laundering looks like in AMLworld.** Verified Part B.1: `src/data/ssl_data.py:36-49` lists eleven 1-hop aggregates plus reciprocity. The AMLworld generator injects eight typologies: fan-in, fan-out, scatter-gather, gather-scatter, simple cycle, random, bipartite, stack. Five of them are multi-hop by construction, and cycles are generally longer than two hops. Reciprocal pairs (`r_2cycle`) are common in ordinary commerce and are the least diagnostic cycle. The 2-step flow entropies already computed in the ETL (`transform_window_convert_currencies.py:183-231`) are closer to scatter-gather and are simply dropped. Also, 2-cycle and fan-in (distinct counterparties) need node identity; the random feature that would supply it is 1-d, frozen at preprocessing (`ssl_data.py:78,117`) and replaced by all-ones downstream since no config sets `encodings_random_feats`. Weakens H1, H3.

**3. Plain GIN/PNA is a strawman on this benchmark.** Verified Part B.8: no reverse message passing, ports or ego IDs, and `emlps` is off in every config. Multi-GNN showed these three adaptations lift GIN on HI-Small from roughly 29 to roughly 57-65 minority F1, and that XGBoost on Graph Feature Preprocessor features reaches roughly 63 (unverified from memory; I could not open the paper). Later work is higher: a 2026 preprint's comparison table, as surfaced by search, lists FraudGT at 69.6 on HI-Small and 28.6 on LI-Small; a multi-edge aggregation line of work (MEGA) reports further gains. A "+Fin-PSE" row that moves GIN from 29 to 40 would be read as rediscovering part of reverse message passing. Undermines the significance of H1.

**4. The handcrafted baseline already exists in the literature, and it is stronger than item 2.** The Graph Feature Preprocessor computes fan/degree/cycle/scatter-gather counts and per-vertex amount and time statistics, fed to gradient-boosted trees. The 12 targets are approximately a subset of its vertex statistics. The honest comparator is therefore GFP+XGBoost and Multi-GNN+GFP features, not "12 statistics into GIN". If Fin-PSE is a lossy neural approximation of features one can compute exactly in a streaming library, H4 (cost) is the only claim left, and exact 1-hop aggregates are cheaper than any GNN forward pass. Threatens H2 and H4.

**5. Pretrain/downstream mismatch makes the current pipeline uninterpretable.** Verified Part B.3, B.4, B.5, B.7. Amounts are USD and z-normalised per 2-day window in pretraining (`ssl_data.py:124`), native-currency and normalised over the whole train period downstream (`aml_data.py:253`). Targets are z-normalised per window (`ssl_data.py:131`) with `-1` sentinels for undefined gaps (ETL line 79), so the encoder learns window-relative ranks of heavy-tailed quantities under L1. `self.encodings.eval()` (`mpnn.py:37`) is undone by `model.train()` (`aml_train.py:92`), so the "frozen" encoder's BatchNorm adapts to downstream data. Ironically that probably rescues it from the input shift, and confounds H3.

**6. Evaluation problems, some inherited.** (a) `add_arange_ids` concatenates an integer id into a float32 tensor (`src/util.py:10-18`); ids above 2^24 are not exact, so on Large (about 180M edges) the `isin` seed mask (`aml_train.py:104,162`) matches non-seed edges. (b) Seed edges whose destination has more than 100 in-edges are dropped from loss and metrics when not sampled; this hits fan-in hubs specifically. I checked the reference implementation: it has the same behaviour for these datasets, so comparability holds but neither is scoring all test edges. (c) Verified B.9: single seed, argmax threshold, epoch 0 not registered as best. Seed variance on LI is known to be several F1 points. (d) `get_optimizer` ignores weight decay for Adam (`util.py:51`), contrary to the plan.

**7. H3 has one real test.** Verified B.10. HI-Medium to HI/LI-Small/Large is the same simulator with different seeds and scales. ETH has no payment format, different time scales and node-level labels on about 1.2k phishing accounts.

## Improvements (value per effort, highest first)

1. **Run the three cheap controls before anything else:** (i) exact targets as node features, computed on the downstream graph; (ii) randomly initialised encoder, frozen and unfrozen; (iii) GFP+XGBoost. One week. If (i) matches Fin-PSE, stop claiming pretraining.
2. **Switch the backbone and reference row to Multi-GIN+EU and Multi-PNA+EU**, keep plain GIN only as a floor. Report against published Multi-GNN, FraudGT and MEGA numbers on Small and Medium HI/LI with 5 seeds.
3. **Give the encoder something the backbone lacks.** Bidirectional sampling and reverse message passing in the encoder, 4-6 layers, wider fan-out than the backbone, and per-batch resampled multi-dimensional random features (or ego IDs) at both stages.
4. **Replace the targets with typology-shaped ones:** temporally ordered k-hop cycle counts (k = 3..6), scatter-gather and gather-scatter counts with time ordering, 2-step flow entropy (already in the ETL), pass-through ratio within a time lag (money in at t, out by t+Δ), amount-conservation along 2-paths. Use log-transformed raw targets with Huber loss, global normalisation.
5. **Fix evaluation:** integer edge ids kept in a separate tensor; force-include seed edges; average precision and recall at fixed alert budget (e.g. precision at top 0.1%) next to F1; threshold tuned on validation.
6. **Make H3 real:** add SAML-D or AMLSim as a second generator, Elliptic2 for subgraph-level labels, and report ETH with pretrain-on-AML vs pretrain-on-ETH vs none.
7. **Add a label-scarcity curve** (1%, 5%, 25%, 100% of training labels). This is where frozen pretrained features plausibly win and where practitioners care.

## The single experiment that would most change my mind

On HI-Small and LI-Small, 5 seeds, with Multi-GIN+EU as the backbone, compare: no encoding; exact 12 statistics as node features; GFP features as node features; random-init frozen encoder; Fin-PSE (after fix 3); all at 100% and at 5% of training labels.

- Fin-PSE beats exact statistics and GFP features by more than one standard deviation at 5% labels, or matches GFP at 100% without any pattern mining: I move to CONTINUE AS PLANNED, because a learned encoder would then be doing real work.
- Fin-PSE is within noise of the exact 12 statistics and below GFP: the pretraining contribution is nil and I move to PIVOT (directions below).
- Nothing helps on top of Multi-GIN+EU: the adaptations already subsume it; DEAD END for the PSE framing.

## Alternative research directions

**A. Typology-masked pretraining (motif completion).** Mask one edge of a temporally ordered k-path, cycle or scatter-gather instance and train the encoder to recover its presence, time bucket and amount bucket. Promising because the pretext task is the laundering structure itself and is label-free. Risk: structures in licit traffic dominate and the task collapses to degree priors. First experiment: masked temporal 3- and 4-cycle closure on HI-Medium, linear-probe on HI-Small with 5% labels versus GFP features.

**B. Cross-generator and cross-institution transfer benchmark.** Train on one simulator or bank partition, test on another (AMLworld, SAML-D, AMLSim; AMLworld split by bank id so each "institution" sees only its own accounts plus boundary edges). Promising because every published AMLworld number is in-distribution and banks never see the full graph. Risk: synthetic-to-synthetic transfer says little about reality. First experiment: Multi-PNA+EU and GFP+XGBoost trained on HI-Medium, tested zero-shot on LI-Small and on a single-bank view.

**C. Flow-conservation message passing.** Treat amounts as flow: messages carry time-ordered cumulative in/out amounts so the network can represent "what entered before t left by t+Δ", the defining property of layering. Promising because no current benchmark model enforces temporal causality along paths. Risk: sampling breaks conservation at hubs. First experiment: add a time-respecting aggregation (only edges earlier than the target edge) to Multi-GIN and measure F1 per typology using AMLworld's pattern files.

**D. Alert-group (subgraph) detection instead of edge classification.** Predict and rank connected laundering subgraphs, evaluated at the investigation-case level, as Elliptic2 frames it. Promising because investigators work cases, not transactions, and a case comes with its own explanation. Risk: ground-truth groupings in AMLworld cover only the patterned minority of illicit edges. First experiment: cluster top-scored edges from an existing model into components and report case-level precision at k against the pattern files.

**E. Neural-symbolic hybrid: learned scoring over mined candidates.** Use fast temporal pattern mining to propose candidates, and a small GNN to score them. Recent throughput-oriented mining work shows the mining side scales but loses accuracy to transformers; closing that gap is open. Risk: recall is capped by the miner's pattern vocabulary. First experiment: GFP candidate edges rescored by Multi-GIN restricted to candidate neighbourhoods; compare F1 and throughput with full-graph inference.
