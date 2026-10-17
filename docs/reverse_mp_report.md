# Reverse message passing and encoding improvements

Branch: `feat/reverse_mp` (from `develop`).

> **Update:** the encoder targets, preprocessing and downstream protocol were redesigned afterwards; see `docs/fin_pse_design.md`. In short:
> - The paper runs use precomputed encodings with the baseline untouched, so reverse MP lives in the encoder only.
> - Currencies are converted with daily rates (`dataset.fx_rates_file` replaces `dataset.currency_rates_file`).
> - The `aml_*_encodings_rmp*.yaml` configs were replaced by `aml_*_finpse.yaml`.
>
> The reverse MP, ports, log and ego options described below are still available.

## Summary

This branch adds reverse message passing (reverse MP) from MEGA-GNN, USD conversion of the AML test graphs, and a few smaller changes that should make the pretrained encodings better. Every feature is off by default and is switched on with a graphgym config key. With all keys at their defaults:
- the code behaves as on `develop`
- the processed dataset files keep their names
- existing encoder and model checkpoints load unchanged

## 1. Motivation

### 1.1 The encoder only sees one direction

fin-pse follows GPSE (Graph Positional and Structural Encoder, arXiv 2307.07107):
1. A `ResGatedGraphConv` encoder (`FinPSEEncoder`) gets random node features.
2. It is pretrained on HI-Medium to predict node statistics.
3. It is frozen and its output is concatenated to the input of the downstream MPNN.

Several of the pretraining targets describe the outgoing side of a node: `deg_out`, `fan_out`, `vol_out`, `mu_gap_out_sec`, `var_gap_out_sec`, `flow_imbalance` and `r_2cycle`. Two things stop the homogeneous model from seeing that side:

1. **Message passing direction.** PyG convolutions use `flow="source_to_target"`, so a node only aggregates messages over its incoming edges. With `k` layers a node's embedding depends only on its `k`-hop in-neighbourhood, so its out-degree and outgoing volume cannot be computed at all.
2. **Neighbour sampling.** `NeighborLoader` and `LinkNeighborLoader` on a homogeneous graph only sample the in-edges of each frontier node. The outgoing edges of a node are usually not even in the batch.

Reverse MP fixes both. The graph becomes a `HeteroData` with a forward edge type `('node', 'to', 'node')` and a reversed copy `('node', 'rev_to', 'node')`. Messages then flow both ways, and the loaders sample along both edge types. On the synthetic smoke-test data, heterogeneous training batches deliver about 23k outgoing-edge messages to the senders of the seed transactions. The homogeneous batches deliver none, by construction.

### 1.2 The test graphs use different currencies from the pretraining graphs

The pretraining graphs (`etl/aml/*convert_currencies*.py`) convert every amount to USD. They use the average yfinance close over the dataset period (`etl/aml/utils.py::convert_currencies_to_usd`). `AMLData` fed the encoder `Amount Received` in its original currency. A Yen or Bitcoin amount therefore looked very different to the encoder than during pretraining. On synthetic data, the z-normalised raw amount correlates only 0.20 with the USD amount.

## 2. What was implemented

### 2.1 Reverse message passing (MEGA-GNN)

| MEGA-GNN | fin-pse |
|---|---|
| `create_hetero_obj`: `rev_to` edges are `edge_index.flipud()`, and port columns are swapped | `src/data/hetero.py::to_hetero_data` |
| `edge_emb_rev` | `MPNN.edge_emb_rev`, `FinPSEEncoder.edge_emb_rev` |
| `to_hetero(GnnHelper, aggr='mean')` | each layer's conv is `HeteroConv({to: conv, rev_to: conv'}, aggr='mean')` |
| per-edge-type edge MLPs (from `to_hetero`) | `ModuleDict` of edge MLPs per edge type in `GnnHelper` |
| hetero `add_arange_ids`, where reverse ids are offset | `src/util.py::add_arange_ids` |
| `LinkNeighborLoader(edge_label_index=(('node','to','node'), ...))` | `src/train/aml_train.py::get_loaders` |
| `--ego` (`AddEgoIds`) | `gnn.ego`, `src/data/hetero.py::AddEgoIds` |
| `--ports` | `dataset.ports`, `src/data/ports.py` |

`to_hetero` builds, for every layer, one copy of the conv per edge type and averages their outputs per destination node. The BatchNorm and residual are shared per node type, and there is one edge MLP per edge type. `HeteroConv(aggr="mean")` produces exactly that structure without FX tracing. A unit test checks that one layer's output equals the mean of the two directional convs followed by BN and the residual.

Checkpoint compatibility:
- Homogeneous parameter names (`convs.i`, `emlps.i`, `batch_norms.i`, `node_emb`, `edge_emb`) are unchanged. An encoder checkpoint written by `develop` loads into the new code and gives bit-identical outputs.
- Reverse MP checkpoints add `edge_emb_rev`, `convs.i.convs.<node___to___node>` / `<node___rev_to___node>`, and `emlps.i.node__to__node` / `node__rev_to__node`.

Two flags control reverse MP, because pretraining and downstream training are separate runs:
- `gnn.reverse_mp` applies reverse MP to the model being trained: `FinPSE` during pretraining, `MPNN` downstream.
- `gnn.encodings_reverse_mp` says that the frozen encoder loaded from `gnn.encodings_file` was pretrained with reverse MP.

Graphs are heterogeneous whenever either flag needs them (`uses_hetero_data()`). This includes an encoder-only setup, because the loaders must sample outgoing edges for the encoder to see them. All four combinations work:

| `encodings_reverse_mp` | `reverse_mp` | behaviour |
|---|---|---|
| false | false | as on `develop` |
| false | true | homogeneous encoder on the forward edges of a hetero batch; the reverse edges reuse the encoder's edge embedding of their raw features |
| true | false | hetero batch; reverse MP encoder; the MPNN uses forward edges only. Isolates the gain from the encodings |
| true | true | reverse MP everywhere (MEGA-style) |

### 2.2 USD conversion of the AML test graphs

`dataset.convert_currencies: true` makes `AMLData.process` multiply `Amount Received` and `Amount Sent` by the USD rate of their currency. The currency id stays as an edge feature, so only the amount changes. The encoder input `[Timestamp, Amount]` is then in USD, as in pretraining.

- Rates reuse the ETL code: same tickers, and the average yfinance daily close over the dataset's own first-to-last timestamp.
- The currency ids of the formatted CSV are rebuilt from the raw CSV in the order `format_dataset` assigns them (receiving currency, then payment currency, row by row). This is checked against `format_dataset`'s output.
- `data/processed/currency_meta_<table>.json` caches the currency ids and the dataset period. `data/processed/currency_rates_<table>.json` caches the rates, so yfinance is only queried once.
- `dataset.currency_rates_file` points to a JSON `{currency name: USD rate}` that is used instead of yfinance. This helps on compute nodes without internet. It also lets you reuse the exact HI-Medium pretraining rates if you want identical multipliers.

### 2.3 Other changes that should improve the encodings

| key | what | why |
|---|---|---|
| `dataset.ports` | Appends the in-port and out-port of every edge (Multi-GNN/MEGA port numbering: a node's neighbours are numbered by first contact) as the last two edge features, per split, before z-norm. Reverse edges get the two columns swapped. | Plain MP cannot tell three transactions from one counterparty apart from one transaction each from three counterparties. Ports make `fan_in`/`fan_out` (unique neighbours) computable and give provably more expressive multigraph GNNs (Egressy et al.). Vectorised O(E log E) numpy code, checked against Multi-GNN's `ports()`, so it scales to AML Large. |
| `gnn.encodings_ports` | The encoder was pretrained with ports and takes `[Timestamp, Amount, in_port, out_port]`. | Needed to reuse a ports encoder downstream; requires `dataset.ports`. |
| `dataset.log_amounts` | `sign(x)·log(1+|x|)` on amount features before z-norm (AML, ETH, SSL). | Amounts span many orders of magnitude. After a plain z-norm almost every transaction sits near the same value and a few outliers dominate. Must be set the same way for pretraining and downstream. |
| `ssl.log_targets` | Same transform on the pretraining targets before z-norm. | `vol_*`, `deg_*` and gaps are heavy-tailed. The L1 loss otherwise mostly learns a constant. |
| `ssl.resample_random_feats` | New `randn` node features for every pretraining batch. | The stored features are fixed per node, so the encoder can memorise HI-Medium nodes. Downstream, `encodings_random_feats` draws fresh noise every forward pass. Resampling matches that, and it is what GPSE does. |
| `gnn.ego` | Ego-ID feature marking seed nodes: endpoints of the seed edges for link loaders, the first `batch_size` nodes for node loaders. Downstream only; the encoder input drops it. | Part of MEGA-GNN's best configurations. Helps the model find cycles through the seed transaction. |
| `gnn.encodings_layers`, `gnn.encodings_dim`, `gnn.encodings_edge_dim` | Shape of the loaded encoder, which was hard-coded to `layers_mp`/64/2. | Resolves the TODO in `fin_pse.py`. A 4-layer encoder can now be used with a 2-layer MPNN. `-1` keeps the old behaviour. |

Processed dataset files get a suffix per enabled option (`_usd`, `_ports`, `_log`, `_logy`, e.g. `data_Small_HI_usd_ports.pt`). With no options the names are unchanged, so existing caches are reused and different variants never overwrite each other.

## 3. Configuration

New example configs:
- `configs/ssl/pretrain_whole_convert_rmp.yaml`: encoder pretraining with reverse MP.
- `configs/ssl/pretrain_whole_convert_rmp_full.yaml`: reverse MP + ports + log amounts/targets + resampled features.
- `configs/aml/aml_{small,large}_hi_{gin,pna}_encodings_rmp.yaml`: reverse MP encoder + reverse MP MPNN + USD conversion + random encoder features.
- `configs/aml/aml_small_hi_pna_encodings_rmp_full.yaml`: downstream counterpart of the `_rmp_full` encoder (ports, log amounts, ego IDs).

Every key can also be overridden from the command line, e.g.:

```bash
# Pretrain an encoder with reverse message passing
python main.py --cfg configs/ssl/pretrain_whole_convert_rmp.yaml

# Use it on AML Small HI, MPNN without reverse MP (isolates the encodings)
python main.py --cfg configs/aml/aml_small_hi_pna_encodings_rmp.yaml \
    gnn.encodings_file <checkpoint>.tar gnn.reverse_mp False

# Offline compute node: use a rates file instead of yfinance
python main.py --cfg configs/aml/aml_large_hi_gin_encodings_rmp.yaml \
    gnn.encodings_file <checkpoint>.tar dataset.currency_rates_file data/rates.json
```

Settings that must match between pretraining and downstream:
- `gnn.reverse_mp` (pretraining) = `gnn.encodings_reverse_mp` (downstream)
- `dataset.ports` (pretraining) = `gnn.encodings_ports` (downstream)
- `dataset.log_amounts` must be equal in both
- the pretraining `gnn.layers_mp`/`dim_inner` must match `gnn.encodings_layers`/`encodings_dim`

## 4. Verification

The container has no GPU, no internet access to Yahoo Finance, and no real AML data. So I tested on CPU with synthetic data in the raw IBM/SSL/ETH CSV formats: 300–500 accounts, 6–9k transactions, mixed currencies, multi-edges.

Unit checks:
- **Checkpoint compatibility.** A `FinPSEEncoder` state dict written by `develop` loads and gives bit-identical outputs. A `develop` MPNN (GIN + edge MLPs) state dict loads.
- **Reverse MP layer.** The `HeteroConv(mean)` layer equals the mean of the two directional convs (then BN and residual).
- **Forward/backward over 24 combinations:** {GIN, PNA} × {reverse_mp on/off} × {no encoder, homogeneous encoder, reverse MP encoder} × {ego on/off}.
- **Ports.** Identical to a Multi-GNN `ports()` reference on 20 random multigraphs, for in-ports and out-ports.
- **Currency conversion.** The rebuilt currency ids equal `format_dataset`'s ids on every row. The converted, normalised amount correlates exactly (r = 1.0) with the expected USD amount.
- **Dataset processing** with and without ports/log/log-target flags for SSL (whole and windowed), ETH and AML: correct shapes, no NaNs, separate cache files.

End-to-end runs of `main.py` (1 epoch, `WANDB_MODE=disabled`), all exit 0:
- SSL pretraining: homogeneous, reverse MP, reverse MP + all extras, windowed reverse MP. The homogeneous run gives exactly the same loss as `develop` (0.349782).
- AML Small HI: baseline; GIN reverse MP with edge MLPs; PNA reverse MP + ports + ego; homogeneous encoder + reverse MP model; reverse MP encoder + homogeneous model; reverse MP encoder + reverse MP model; full configuration with a ports encoder.
- AML Large HI with the new config.
- ETH: reverse MP + ports + ego; reverse MP encoder.
- Default configs match `develop` within run-to-run noise: ETH PNA and AML PNA with encodings to 1e-10; the AML GIN baseline to 1e-5, which is also its run-to-run variation from CPU scatter nondeterminism.

Not verified here:
- The live yfinance call (Yahoo is blocked in this container). The code is the ETL function; only the rates-file path was exercised.
- GPU runs and real-size data. For AML Large, port numbering keeps a handful of int64 arrays of length E in memory, a few GB for ~180M edges, per split.
- Model quality. The synthetic runs only prove the code paths work. Real numbers need cluster runs.

## 5. Suggested experiments

1. Pretrain with `pretrain_whole_convert.yaml` and with `pretrain_whole_convert_rmp.yaml`. Compare the per-target validation loss, especially for the `*_out` targets. Logging a per-target loss would make this much clearer than the single average.
2. Downstream on AML Small/Large HI/LI:

   | run | encoder | `reverse_mp` | `convert_currencies` |
   |---|---|---|---|
   | baseline | none | off | – |
   | old encodings | homogeneous encoder | off | off |
   | old encodings, USD test graphs | homogeneous encoder | off | on |
   | reverse MP encoder only (answers "do reverse MP encodings help?") | reverse MP encoder | off | on |
   | MEGA-style reverse MP, no encodings | none | on | – |
   | both | reverse MP encoder | on | on |

3. Then add `ports`/`log_amounts`/`log_targets`/`resample_random_feats` (the `_rmp_full` pair), one at a time if budget allows.

## 6. Observations and caveats

- **Seed-edge coverage (existing behaviour).** Loss and metrics are computed only on the seed edges that `LinkNeighborLoader` happens to sample into their own batch (`seed_edge_mask`). On the synthetic data with 10 neighbours this was 50–77% of the seeds, with the fraction differing between splits. It should be much higher with `[100, 100]` on real data, but it is worth logging. MEGA-GNN adds missing seed edges back for some datasets.
- **Encoder edge output.** `FinPSEEncoder` returns `edge_emb(raw edge features)` as edge encodings. Edges are not updated by message passing, so `project_edge_encodings` only adds a re-embedding of `[Timestamp, Amount]`. Encodings built from the endpoint node encodings, e.g. `[h_src, h_dst]`, would carry more structure.
- **Random features.** Pretraining uses random node features, but `gnn.encodings_random_feats` defaults to `false`, so the downstream encoder gets all-ones features by default. The new configs set it to `true`.
- **Normalisation mismatch.** Pretraining z-normalises the encoder inputs per SSL graph (per window in the windowed variants). Downstream normalises with the train split statistics. Timestamps especially end up on different scales. Consider pretraining on whole-graph data, or storing the pretraining statistics and reusing them downstream.
- **PNA degree histogram.** It is computed on forward in-degrees and shared by both directions, as in MEGA-GNN.
- **Not ported.** MEGA-GNN's main contribution, two-stage multi-edge aggregation (`--flatten_edges --edge_agg_type`), was not ported. It is the natural next step on top of reverse MP and ports. GPSE-style deeper encoders and a virtual node are also open.
- **Parameters.** Reverse MP doubles the conv and edge MLP parameters of every layer:
  - encoder (64 dims, 2 layers): 58k to 116k
  - MPNN GIN (64 dims): 37k to 62k
  - MPNN PNA (20 dims): 29k to 53k

  If capacity rather than direction explains a gain, a homogeneous model of matching size is the fair control.

## 7. Tooling notes

- I installed the caveman skill (JuliusBrussee/caveman) and used it for chat replies only. Code, commits and this report are normal prose.
- codebase-memory-mcp could not be used. It was not in the environment. Its release binary needs the GitHub API (blocked), and building it from source was blocked by the session's permission policy. Add it to the cloud environment's setup script to use it in a future session.
