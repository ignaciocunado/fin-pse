# Fin-PSE: encoder targets, preprocessing and downstream protocol

Branch `feat/reverse_mp`. This document supersedes the downstream parts of `docs/reverse_mp_report.md`.

## 1. Aim

The paper claims that a plain GNN baseline (GIN/PNA MPNN) gets better AML and phishing results when you add encodings from a small encoder that was pretrained on finance-specific tasks and then frozen (GPSE-style). The protocol below is built to test exactly that claim. The baseline stays identical in every arm; the only thing that changes is whether node encodings are added.

## 2. Pipeline

```
raw IBM AML CSV ──► etl/aml/build_ssl.py ──► <name>_SSL_{Nodes,Trans}_finpse.csv
                   (daily USD, crops, targets)        │
                                                      ▼
                             main.py --cfg configs/ssl/pretrain_fin_pse.yaml
                             (4-layer reverse-MP encoder, per-target R², checkpoint + metadata)
                                                      │
                                                      ▼
downstream raw CSV ──► AMLData / ETHData ──► encodings precomputed once per split graph
                       (baseline features unchanged)   (src/networks/encodings.py, cached)
                                                      │
                                                      ▼
                         main.py --cfg configs/aml/aml_*_finpse.yaml   (baseline + encodings)
                         scripts/transfer_r2.py                        (does the encoder transfer?)
```

## 3. Decisions and why

| Decision | Choice | Why |
|---|---|---|
| Baseline | Unchanged MPNN, features, `[100, 100]` sampling, batch 8192, 100 epochs. No reverse MP, ports or ego IDs. | The claim is about adding encodings, so nothing else may change. The sampling and batch settings are the defaults of Multi-GNN (Egressy et al.) and MEGA-GNN. |
| How encodings enter | Precomputed per split graph, node encodings only, concatenated to the node embeddings | The baseline batches stay bit-identical. Every account is encoded under pretraining conditions, i.e. as a seed with its own sampled neighbourhood. In-batch encoding gives frontier nodes truncated neighbourhoods. The earlier "edge encodings" were only a linear re-embedding of the raw edge features, and with USD conversion they would have leaked converted amounts into the encodings arm. |
| Direction | Reverse MP inside the encoder only | Every out-side signal (fan-out, dwell, onward flow, where money goes) needs it. The encoder is a feature extractor, so this does not change the baseline. Controls separate "direction" from "pretraining". |
| Pretraining data | HI-Medium | Contains more laundering typologies; labels are never used. Downstream graphs are different simulations. |
| Currencies | Two independent switches (§5.1). Encoder: daily USD (default) or native per-currency targets. Baseline: daily USD in every arm (`dataset.convert_currencies: true`). Rates are the daily close of each currency against USD on the transaction's date, from one shared table (`data/fx_daily_usd.csv`). | As accurate as possible, and identical everywhere they are used. Weekend and holiday gaps are forward-filled. Native mode never adds up amounts of different currencies. |
| Span | Whole graph, plus prefix crops (20/40/60/80%), plus 5 random 2–7 day intervals. Targets are recomputed per crop and time locality is built into the target definitions. | Downstream split graphs are prefixes. Per-window graphs would not match the static encodings computed on 11–60+ day graphs. |
| Time scale | Days since the start of the graph, normalised with fixed pretraining statistics | A "1-day gap" must mean the same in every dataset. Per-graph z-norm breaks that. |
| Amounts | USD (or per currency in native mode), signed log | Amounts are heavy-tailed. |
| Normalisation | Statistics over all pretraining graphs, stored in the checkpoint and reused downstream | Per-window normalisation shifted the inputs between graphs. |
| Node features | Random, resampled per batch; downstream averaged over 4 draws | Breaks symmetry (fan, cycles) without the encoder memorising accounts. |
| Targets | Four switchable groups: ego, temporal, flow, motifs (§4) | Covers what the paper promises: flow-aware reachability, temporal patterns, crime structures. |
| Motifs | Scatter-gather and temporal cycles from IBM's Graph Feature Preprocessor (GFP, snapml), plus own 2-cycles and gather-scatter | GFP is exact and established. It is also the obvious competitor, so the "exact features" control includes it. |

## 4. Targets (`etl/aml/targets.py`)

All targets are per account and per graph. Missing values are NaN and the loss skips them. All are put through signed log and z-normalised for training.

**ego**
- `deg_in/out`: number of transactions.
- `fan_in/out`: number of distinct counterparties.
- `vol_in/out`: USD volume.
- `flow_imbalance`: (in − out)/(in + out).
- `max_amt_in/out`, `amt_cv_in/out`: largest amount and coefficient of variation per direction. A low CV means many similar amounts (structuring).

**temporal**
- `mu/var_gap_in/out_sec`: mean and variance of the time between consecutive transactions.
- `dwell_median/mean_sec`: from each incoming transaction to the next outgoing one.
- `pass_through_24h`: share of incoming transactions followed by an outgoing one within 24h.
- `peak_day_{deg,fan,vol}_{in,out}`: busiest calendar day.
- `burstiness_in/out`: peak day / mean active day.
- `active_span_days`.

**flow** ("where does the money go, and how much"). With P[i, j] the share of i's outgoing USD sent to j:
- `H/supp/pmax_out_{1,2}_amt`: entropy, number of destinations and largest destination share of P^k[i, ·] (k = 1, 2).
- `H/supp/pmax_in_{1,2}_amt`: the same for where i's money comes from.
- `return_{2,3,4}`: P^k[i, i], the share of i's money back at i after k steps. k = 2 is exact; k = 3, 4 use 64 amount-weighted random walks.
- `sink_share_{1,2}`: share ending within k steps at accounts that keep at least 90% of what they receive.
- `onward_share_24h`: share of i's outgoing money that the recipients send on within 24h. This one is time-respecting.

Switch the group off by removing `flow` from `ssl.target_groups`. A pairwise objective ("share of i's money reaching j") is left for later.

**motifs**
- `r_2cycle`: reciprocal transactions.
- `gather_scatter_daily`: max over days of min(distinct senders, distinct recipients).
- GFP: `scatter_gather_in/out` (patterns an account's incoming/outgoing transactions take part in) and `temporal_cycles`.
- Simple cycles (`lc-cycle`) are off by default because their enumeration explodes in dense neighbourhoods. Enable with `--gfp-params '{"lc-cycle": true}'`.

**Native mode** (encoder trained without currency conversion, §5.1). Amounts of different currencies are never added up:
- The 27 amount targets (volumes, flow imbalance, max/CV of amounts, busiest-day volume and every flow target) are computed **within each currency**:
  - incoming money: `Amount Received` in `Receiving Currency`
  - outgoing money, and the flow that follows it: `Amount Paid` in `Payment Currency`, along same-currency paths only
- New groups:
  - `per_currency`: `<target>__<Currency>` for a fixed vocabulary of the 15 AML currencies and Ether (432 columns). NaN where an account never uses the currency.
  - `main_currency`: `<target>__main`, in the currency of most of the account's transactions.
  - `currency`: number of currencies, currency entropy and top-currency share per direction, by transaction count.
- `ego`, `temporal` and `flow` keep only their count-, time- and support-based targets. `motifs` is unchanged.
- In total 492 targets (489 without GFP).

Every target was checked against a brute-force computation on random graphs (native mode on graphs with mixed and cross-currency transactions). The Monte-Carlo return mass is within 0.004 of the exact P³ and P⁴. The GFP aggregation was checked on hand-built cycles and scatter-gather patterns.

## 5. Preprocessing

| | Encoder (pretraining = downstream) | Baseline model |
|---|---|---|
| Amount | USD mode: Amount Received × daily USD rate. Native mode: amount received + one-hot receiving currency, amount paid + one-hot payment currency. Signed log, pretraining z-stats. | Amount Received × daily USD rate (`dataset.convert_currencies: true`) + currency id, train-split z-norm |
| Time | Days since graph start, pretraining z-stats | Unchanged |
| Node input | Random, resampled; 4-draw average at precompute | Constant 1 |
| Direction | Reverse MP | Homogeneous |
| Sampling | Encoder's own, from the checkpoint (`[500, 20, 10, 5]` suggested) | `[100, 100]` LinkNeighborLoader |

### 5.1 Currency switches

| Where | Switch | Effect |
|---|---|---|
| Encoder (pretraining data) | `build_ssl.py` (default) vs `build_ssl.py --no-convert` | Daily-USD targets vs per-currency targets (files `*_finpse.csv` vs `*_finpse_native.csv`) |
| Encoder (pretraining) | `ssl.convert_currencies: true` / `false` | Matching encoder inputs. `configs/ssl/pretrain_fin_pse.yaml` / `pretrain_fin_pse_native.yaml` |
| Encoder (downstream) | none, follows the checkpoint metadata | The precompute builds USD or native inputs exactly as in pretraining. One processed dataset serves both. |
| Baseline model | `dataset.convert_currencies` | `true` in all AML `*_finpse.yaml` configs: every arm, including the baseline, sees daily-USD amounts. ETH has a single currency. |
| Exact control | `gnn.exact_convert_currencies` | USD or per-currency exact targets. For per-currency, also add `currency`, `main_currency`, `per_currency` to `gnn.exact_target_groups`. |

The baseline arm uses the same config with `gnn.add_encodings False`, so it also gets converted amounts. Its numbers are therefore not directly comparable with the published Multi-GNN/MEGA results, which use raw amounts. Set `dataset.convert_currencies False` to reproduce those.

Native mode on ETH maps Ether to its own vocabulary slot, which the AML-pretrained encoder never saw. Expect weaker transfer there than in USD mode.

## 6. Running it

```bash
# 1. Pretraining data (from the repo root; yfinance fills data/fx_daily_usd.csv on the first run)
python etl/aml/build_ssl.py --input data/raw/HI-Medium_Trans.csv --name HI-Medium --implied-rates

# 2. Pretrain (saves <checkpoint_dir>/HI-Medium_SSL_finpse_epoch_30_<time>.tar with head + metadata)
python main.py --cfg configs/ssl/pretrain_fin_pse.yaml

# 2b. Alternative encoder without currency conversion (per-currency targets)
python etl/aml/build_ssl.py --input data/raw/HI-Medium_Trans.csv --name HI-Medium --no-convert
python main.py --cfg configs/ssl/pretrain_fin_pse_native.yaml

# 3. Downstream arms, all with the same baseline model
CFG=configs/aml/aml_small_hi_gin_finpse.yaml; ENC=gnn.encodings_file <checkpoint>.tar
python main.py --cfg $CFG gnn.add_encodings False                  # baseline
python main.py --cfg $CFG $ENC                                    # + pretrained encoder
python main.py --cfg $CFG $ENC gnn.encodings_source random_init   # + same encoder, untrained
python main.py --cfg $CFG gnn.encodings_source exact              # + exact targets (incl. GFP)
python main.py --cfg $CFG gnn.encodings_source exact gnn.exact_convert_currencies False \
    gnn.exact_target_groups "['ego','temporal','flow','motifs','currency','main_currency','per_currency']"
python main.py --cfg $CFG gnn.encodings_file <no-rmp checkpoint>.tar   # + encoder pretrained with gnn.reverse_mp False
python main.py --cfg $CFG gnn.add_encodings False gnn.reverse_mp True   # optional: reverse MP in the model

# 4. Does the encoder still predict its targets on the downstream graph?
python scripts/transfer_r2.py --cfg $CFG $ENC
```

Encodings are cached under `data/processed/encodings/` per split, source, checkpoint and seed. Cached and fresh runs give the same training run, because the precompute does not touch the global RNG. Use ≥ 5 seeds (`--repeat 5`) and report minority-class F1, mean ± std.

### New configuration keys

| Key | Meaning |
|---|---|
| `ssl.target_groups` | Target groups to predict; empty = original 12 targets |
| `ssl.group_weights` | Loss weight per group |
| `ssl.val_fraction` | Held-out accounts per graph |
| `ssl.max_seeds_per_graph` | Training seeds per graph per epoch (0 = all) |
| `ssl.time_in_days` | Timestamps as days since graph start |
| `gnn.encodings_precompute` | Precomputed encodings (vs the legacy in-batch encoder) |
| `gnn.encodings_source` | `pretrained` / `random_init` / `exact` |
| `gnn.encodings_draws` | Random-feature draws averaged per account |
| `gnn.encodings_batch_size` | Precompute batch size |
| `gnn.exact_target_groups` | Target groups used by the `exact` source |
| `gnn.exact_gfp` | Include the GFP targets in the `exact` source |
| `dataset.fx_rates_file` | Path of the daily rates table |
| `ssl.convert_currencies` | Encoder in daily USD (`true`) or native per-currency mode (`false`) |
| `gnn.exact_convert_currencies` | USD or per-currency targets for the `exact` source |

## 7. Verification

The container has no GPU, no real data and no Yahoo access, so I tested on CPU with synthetic data in the IBM/ETH formats:

- **FX**
  - Weekend forward-fill, cache reuse and extension, and the 23:59 vs 00:01 day boundary behave correctly.
  - ETL conversion and `AMLData` conversion agree exactly (r = 1.0).
- **Targets:** all 51 USD targets match brute force. Native mode was checked on graphs with mixed and cross-currency transactions: per-currency amounts, flow and Monte-Carlo return mass, main-currency selection and currency mix (§4).
- **Pretraining**
  - 10 crops, 50 targets (USD) or 492 targets (native), per-target R² logged. The checkpoint stores the head and the metadata.
  - The legacy pretraining loss is unchanged (0.349782). After the native-mode changes the processed USD data is bit-identical and the USD loss unchanged (0.579152). A run that processes the data now trains exactly like a cached run.
- **Downstream**
  - Baseline edge features, labels and the first training batch (`n_id`, `e_id`, `edge_attr`) are bit-identical with and without encodings.
  - Precomputed encodings equal a full-graph pass when sampling covers the full neighbourhood.
  - Cached and fresh runs agree to 1e-8.
  - Every arm runs end to end on AML and ETH:
    - baseline
    - pretrained with a USD and with a native encoder
    - random_init
    - exact with and without GFP, in USD and per-currency mode
    - reverse MP in the model
  - The baseline stays bit-identical across arms with `dataset.convert_currencies` on, and its converted amounts match the daily rates (r = 1.0).
  - `scripts/transfer_r2.py` works for both encoder modes. It skips per-currency targets with too few accounts or no spread on the downstream graph.
  - A checkpoint without metadata is rejected.
  - Legacy configs still run.
- **Static checks:** `black --check` and Python 3.9 syntax pass.

**One finding changed the evaluation.** Validation R² pooled over all crops was far above the R² within the whole graph: on synthetic data, deg_in was 0.72 pooled vs −1.8 within the whole graph. Most of the target variance lies *between* crops of different sizes. Pretraining therefore now reports R² within each graph, averaged over graphs, plus the whole graph separately. `scripts/transfer_r2.py` reproduces the within-graph R² of the same graph.

**Not verified here:**
- live yfinance
- GFP at HI-Medium scale
- GPU memory with 4-hop reverse-MP sampling
- any real model quality

## 8. Caveats and open issues

- **Simulator vs real rates.** AMLworld generates amounts with its own exchange rates. Daily yfinance rates are faithful to the real world but maybe not to the simulator. `build_ssl.py --implied-rates` reports the discrepancy per currency pair; worth a sentence in the paper.
- **GFP cost.** Scatter-gather is fast. Temporal cycles took over 5 minutes on a dense synthetic graph and seconds on a sparse one. On HI-Medium, run once with the default windows. If it is slow, shorten `temp-cycle_tw` via `--gfp-params`.
- **Sampling for a 4-layer reverse-MP encoder.** Each hop samples per direction. `[500, 20, 10, 5]` keeps hub degrees observable; tune the batch size to GPU memory.
- **Exact control and transfer.** Both compute the targets on each downstream split graph, which costs extra time on AML Large.
- **ETH.** Timestamps must be Unix seconds (checked). ETH-USD history on yfinance starts in late 2017; earlier days take the first available close and a warning is logged.
- **License.** snapml is proprietary (free to use). It is only needed for the GFP motif targets and the exact control.
- **Legacy paths remain** (in-batch encoder, the original 12 targets, MPNN reverse MP / ports / ego) for reproducing earlier runs.

## 9. Paper notes

- Pretraining dataset: HI-Medium (the draft says Medium-LI).
- Encoder: 4 ResGated layers with reverse message passing (the draft says 4 layers; the old configs used 2).
- Drop "from masked neighbourhoods": nothing is masked, and masking would make training differ from inference.
- Describe the flow targets (§4) as the "flow-aware reachability" objective. The time-local targets (dwell, pass-through, busiest day) cover "temporal bursts".
- Controls:
  - random-init encoder: is pretraining needed?
  - exact targets incl. GFP: is learning needed? (Blanuša et al. is cited)
  - encoder without reverse MP: is direction needed?
- Report transfer R² (encoder targets on the downstream graphs) and within-graph pretraining R².
