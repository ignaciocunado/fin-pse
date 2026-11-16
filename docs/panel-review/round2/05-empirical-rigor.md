# Rebuttal 05: empirical rigour and code audit

Checks were source reads plus one small script (PyG 2.8.1; the project pins an older version). The loaders still could not be run, so sampler claims remain source-verified only.

## Fact-check of other panellists' claims

| Claim | Who | Result | Evidence |
|---|---|---|---|
| Edge ids are stored as float32 and collide above 2^24, so the seed mask matches non-seed edges on Large | 02 | **Holds; I missed it** | `src/util.py:18` concatenates an integer `arange` with float features; I confirmed the result is float32. At 1.5×10^8, 32 consecutive ids collapse to 3 distinct values. HI-Large (about 180M edges per 02 and 06, not checked by me) is affected, and so is any table above 16.7M edges. |
| The `ResGatedGraphConv` gate is ReLU, not sigmoid | 01, 03 | **Holds; I missed it** | `fin_pse.py:33` passes `act=`; in the library that argument is the gate (`self.act(k_i + q_j) * v_j`, default `Sigmoid`). |
| Edge encoding is a frozen rank-2 map | 01, 06 (and me) | **Holds** | `fin_pse.py:40,47`: `Linear(2, 64)`; its image is a 2-d affine subspace. |
| Fin-PSE as implemented costs about 10^16 MACs on HI-Large | 06 | **Order of magnitude holds; the per-edge figure is wrong and too low** | The batch arithmetic checks out (about 22K batches per epoch, 2.2M in total). But there is no 64×64 `lin_edge`: with `edge_dim` set, key, query and value are each `Linear(128, 64)` applied per edge, about 25K MACs per edge per layer, six times 06's 4K. The 10^6 sampled edges per batch is unmeasured. The total is more likely 10^17. |
| ETL runs in "16 minutes" | 06 | **Overstated** | `format_ssl_data.sbatch:5` is a requested time limit, not a measurement. |
| Timestamp unit is about 0.58 days in pretraining; mismatch of a few times on AML, two orders of magnitude on ETH | 03 | **Holds** | 2 days / √12 = 0.58 days. The ETH span is 1,261 days (`etl/eth/eda.ipynb`), so the training-split standard deviation is hundreds of days. |
| PNA "computes in-degree, in-volume and time spread exactly in its first layer" | 04 | **Wrong for degree and volume** | The configured aggregators are mean, min, max, std (`mpnn.py:111`), with no sum. Degree enters only as a log scaler, and sampling caps it at 100. Time spread via max and min is right. |
| BatchNorm drift "probably rescues" the encoder from input shift | 02, 04 | **Unsupported** | The BatchNorm sits after each convolution (`fin_pse.py:43-44`). The shifted inputs first pass a linear embedding and an unbounded ReLU-gated product, which an affine re-standardisation afterwards does not undo. |
| Dropped seed edges match the reference implementation, "so comparability holds" | 02 | **Overstated** | The behaviour may be shared (I did not open the reference code), but the scored subset depends on the sampler seed and on the id collisions above. |

## Where I now agree

- **01 (containment).** The right capacity control is a 4-layer backbone on the same subgraph, not my parameter-matched 2-layer one. I add that arm to my gate.
- **02 and 01 (backbone).** My gate used plain GIN. The gate must also run on a reverse-message-passing backbone, because a plain directed baseline is repaired by any out-degree feature.
- **03 (time).** Relative log time with no per-split z-scoring is a better fix than my "use one convention in both phases".
- **01 and 04 (pretext audit).** Per-target held-out R², plus a ridge regression of out-targets on in-targets, is cheaper than anything I proposed and should come first.
- **Concessions.** I missed the float32 id bug and the ReLU gate. My note that the mask is correct holds only below 2^24 edges. My claim of "no label leakage through the seed mask" stands, since the collisions corrupt which edges are scored and not what the model sees.

## Where I disagree

**04: PIVOT now, and "H2 is false by construction".** It is not by construction. A hidden state trained on 12 targets can carry more than the 12 numbers, and whether it does is a one-week measurement once the harness is fixed. 04's replacement (masked edge-attribute modelling across many ledgers) is as unvalidated as the current objective and would be run on the same broken evaluation. Pivoting before measuring swaps one untested idea for a larger one.

**02: report against published Multi-GNN, FraudGT and MEGA numbers.** The figures in 02 are from memory or search snippets. More importantly, numbers produced by a harness that scores a sampler-dependent subset of test edges are not a fixed target. Every comparator must be rerun in the corrected harness; published numbers belong in a footnote.

**All five: "5 seeds, more than one standard deviation".** With five seeds, a one-standard-deviation gap between two arms occurs by chance far too often, and LI-Small variance is several F1 points by 02's own account. I hold to ten seeds, paired tests and average precision as the primary metric. The low-label arms that 02 and 04 propose have only tens to low hundreds of positives on Small and need more seeds still.

**03: the time-shuffle control first.** It is a good experiment, but uninterpretable until metrics, seed scoring and encoder freezing are fixed.



## Open question for the panel

Should the decisive control be the exact statistics on the plain backbone or on a reverse-message-passing backbone? On the plain backbone the exact statistics supply out-direction information the model cannot otherwise see, so they will win for a reason unrelated to pretraining. On the adapted backbone nothing may help at all. The author needs the panel to agree which result ends the PSE framing.

## Final verdict

**CONTINUE WITH MAJOR CHANGES, confidence medium. Unchanged**, though the cost fact-check and 01's containment argument move me closer to PIVOT than I was.

Probability that the exact 12 statistics as node features match or beat frozen Fin-PSE: **0.8** (0.85 for the code as implemented, about 0.75 after reverse edges and precomputation).

## Ranked ballot of next steps

1. Repair the harness: integer ids in a separate tensor, score every seed edge, softmax-based average precision, seeds logged to a results file (05, 02).
2. Run the pretext audit: held-out per-target R² and the in-to-out ridge baseline (01, 04).
3. Run the gate on HI-Small and LI-Small, ten seeds, on both plain and reverse-message-passing backbones: none, exact 12 statistics, random frozen encoder, Fin-PSE, 4-layer backbone (all six papers).
4. Add reverse edges, relative log time, and precompute encodings once with the encoder locked in eval mode (03, 06, 05).
5. Swap in expensive targets, starting with the unused 2-step flow entropy and temporal 3- and 4-cycle counts (01, 02, 04, 06).

## Ranked ballot of alternative directions

1. **Exact and sketched temporal-motif features with learning on top, on a leakage-audited harness** (06 direction 2 merged with my direction 1 and 02's direction B). It is the lowest-risk path, it produces the baseline every other direction must beat, and the audit findings are a contribution on their own.
2. **Amortised temporal-motif encoder** (01 direction 2, 04 direction 3, 02 direction A, my direction 2). This is the only version of the original idea where amortisation has value, and it has a clear failure test in held-out count R².
3. **Hodge edge-flow encodings** (06 direction 1). It is cheap and falsifiable in one experiment, but it depends on the simulator conserving flow realistically, which nobody has checked.

**Advise against: multi-ledger masked-event pretraining** (04 direction 1). It multiplies data, compute and evaluation surface before a single trustworthy number exists, and its best evidence is transfer among token networks, not laundering detection.
