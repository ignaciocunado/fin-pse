# Panel brief: Fin-PSE

This brief is shared by every panellist. Part A is the author's research plan, verbatim.
Part B is a list of preliminary observations the coordinator made while reading the code.
Part B is **unverified**: check anything you rely on against the source, and say so if it is wrong.

---

## Part A. Research plan (author's own words)

### One-line summary

Pretrain a positional/structural encoder on financial transaction graphs using finance-specific, self-supervised targets, then freeze it and feed its output into standard GNNs (GIN, PNA) as extra features. The aim is to see whether this improves financial-crime detection and whether the encoder transfers across financial graphs.

### Background and motivation

- **Data model.** Accounts are nodes. Transactions are directed edges with a timestamp and an amount (and, for AML, currency and payment format).
- **Why GNNs.** Money laundering and phishing show up in graph structure and in behaviour: reciprocal transfers (2-cycles), bursts of activity, fan-in/fan-out, and how money flows through a node.
- **The gap.** Existing positional and structural encodings (LapPE, RWSE, GPSE, Graphormer-style distances) were built for small, mostly undirected molecular graphs. They ignore edge direction, timing, and money flow, and many don't scale to large graphs. GPSE, for example, assumes undirected graphs.
- **The idea.** Learn a node encoding that is aware of these financial semantics and can be reused across tasks, rather than hand-engineering features for each dataset.

### Method, as implemented

#### 1. Self-supervised pretraining (`src/train/ssl_pretrain.py`, `src/data/ssl_data.py`)

- **Model.** `FinPSEEncoder` (`src/networks/fin_pse.py`): a linear node embedding and a linear edge embedding to 64 dimensions, then 2 residual `ResGatedGraphConv` layers with BatchNorm and ReLU. Each layer uses the update `x = (x + relu(BN(conv(x)))) / 2`.
- **Inputs.** Node features are random Gaussian noise (`x ~ N(0,1)`), so the encoder learns structure rather than identity. Edge features are `[Timestamp, Amount]`, z-normalised. Pretraining uses USD-converted amounts.
- **Targets.** For each node, 12 z-normalised statistics:
  - Mean and variance of incoming and outgoing inter-arrival times (4)
  - In-degree and out-degree (2)
  - Fan-in and fan-out, meaning the number of distinct counterparties (2)
  - Incoming and outgoing volume (2)
  - Flow imbalance (1)
  - Reciprocal (2-cycle) count (1)
- **Objective.** L1 loss between predicted and true targets for the seed nodes of each batch.
- **Training.** Adam, learning rate 0.003, weight decay 1e-4, 30 epochs, batch size 512. Each batch is built with `NeighborLoader` using 100 neighbours per hop.
- **Data.** IBM AML HI-Medium, in windowed (2-day) and whole-graph variants, with and without currency conversion. Windowed versions are used so that time-based leakage is limited.
- **Output.** A checkpoint of the encoder's `state_dict`. The pretraining objective is not used downstream.

#### 2. Plugging the encoder into a downstream GNN (`src/networks/mpnn.py`)

- The pretrained encoder is loaded and **frozen**. It is run under `no_grad` on a copy of the input graph.
- Its node outputs (64-d) and edge outputs (64-d) each go through a projection MLP (Linear, ReLU, BatchNorm, Linear) to `dim_inner / 2`.
- These projections are concatenated with the backbone's own node and edge embeddings, so the backbone sees both.
- The backbone is either GIN (`GINEConv`) or PNA (`PNAConv` with mean, min, max, std aggregators), with 2 message-passing layers and `dim_inner` 64 (20 for ETH).
- An optional `encodings_random_feats` flag replaces the node inputs with noise at downstream time, to match pretraining.

#### 3. Downstream tasks

| Task | Dataset | Level | Splits | Loss | Model selection |
|---|---|---|---|---|---|
| AML laundering detection | IBM AML Small-HI, Small-LI, Large-HI, Large-LI | Edge classification (illicit vs licit transaction) | Chronological, roughly 60/20/20 by day, chosen to balance transaction share and illicit ratio | Cross-entropy, class weights 1:6 | Best validation F1 |
| Phishing detection | Ethereum phishing (ETH) | Node classification | Chronological by each account's first timestamp, at the 65th and 80th percentiles | Cross-entropy, class weights 1:6 | Best validation F1 |

- **Metrics.** F1, precision, recall, and PR-AUC. PR-AUC is computed with `auc(recall, precision)`.
- **Baselines in the code.** Each backbone runs with and without `add_encodings`. Configs such as `aml_small_li_gin.yaml` and `aml_small_li_gin_encodings.yaml` show the pairing.

### Hypotheses

- **H1.** Frozen Fin-PSE features improve F1 and PR-AUC for GIN and PNA on the AML and ETH tasks.
- **H2.** The gain comes from the finance-specific targets, not from extra parameters or generic pretraining. This requires ablations and capacity-matched baselines.
- **H3.** The encoder transfers across datasets and generators, for example from AML pretraining to ETH.
- **H4.** Fin-PSE is cheaper than LapPE, RWSE, and GPSE on large graphs, and it handles direction and time natively.

### Evidence needed (author's list)

1. **Main table.** GIN and PNA, with and without Fin-PSE, on AML Small/Large (HI and LI) and ETH. At least 5 seeds, mean ± std.
2. **Handcrafted-feature baseline.** Feed the 12 target statistics directly as node features. If it matches Fin-PSE, the pretraining adds little beyond feature engineering, and the contribution has to rest on scalability or transfer.
3. **Random or untrained encoder.** Same architecture, not pretrained.
4. **Existing PSEs.** Degree-only, RWSE, LapPE, GPSE, and possibly a directed spectral variant, where feasible.
5. **Ablations.** Each objective alone and combined. Windowed vs whole-graph. Currency conversion on vs off. Encoder depth. Frozen vs fine-tuned.
6. **Efficiency.** Pretraining time, per-epoch overhead, and memory on Small, Medium, and Large.

---

## Part B. Coordinator's preliminary code observations (UNVERIFIED, check before relying on them)

The repository contains no results, no tests, and a one-line README. Status: pretraining and downstream pipelines are implemented; none of the "evidence needed" baselines (items 2-4) exist in code yet.

1. **Targets look mostly 1-hop.** Eleven of the 12 targets are aggregates over a node's own incident edges (degree, distinct counterparties, volume, gap mean/variance, imbalance). Only `r_2cycle` needs reciprocity information. GPSE's targets, by contrast, were multi-hop or global (LapPE, RWSE, electrostatic, heat-kernel, cycle counts). The ETL (`etl/aml/transform_window_convert_currencies.py`) also computes 2-step flow-entropy features (`H_out_2_amt`, etc.) that are not in the target list in `src/data/ssl_data.py`.
2. **Out-direction targets may be unobservable.** PyG `NeighborLoader` samples edges pointing *into* seed nodes, and `ResGatedGraphConv` passes messages source to target. There is no reverse message passing. If so, a seed node's out-degree, fan-out, out-volume, and outgoing gap statistics are not in its computational graph at all. Verify against PyG semantics.
3. **Random node features are fixed, 1-dimensional, and mismatched downstream.** `ssl_data.py` draws `torch.randn((num_nodes, 1))` once at dataset-processing time, so it is stored and reused every epoch. Downstream, node features are all-ones unless `encodings_random_feats` is set, so the frozen encoder sees an input distribution it was never trained on.
4. **Edge-feature distribution shift between pretraining and downstream.** Pretraining: `[Timestamp, Amount]` with amount converted to USD, z-normalised per window. Downstream (`mpnn.py`): the first two columns of the downstream edge features, i.e. `[Timestamp, Amount Received]` in native currency, z-normalised with train-split statistics over the whole period. ETH uses its own `[timestamp, amount]` scale.
5. **Frozen encoder's BatchNorm may not be frozen.** `self.encodings.eval()` is called in `__init__`, but the training loop calls `model.train()`, which would put the submodule back into training mode and update BN running statistics on downstream data.
6. **Encoder depth is tied to the backbone.** `FinPSEEncoder` reads `cfg.gnn.layers_mp`, so the downstream config determines how many encoder layers are built before loading the checkpoint.
7. **Targets are z-normalised per window**, and gap statistics for nodes with fewer than two transactions are filled with `-1` before normalisation.
8. **Backbones are plain directed GIN/PNA.** No reverse message passing, port numbering, or ego IDs (the Multi-GNN adaptations for this benchmark). Edge updates (`emlps`) exist as an option.
9. **Evaluation.** Model selection on validation F1 at a fixed argmax threshold; test is evaluated every epoch; single seed unless `--repeat` is passed; `best_val_f1` is not updated on epoch 0.
10. **Pretraining data and downstream data come from the same simulator** (IBM AMLworld), HI-Medium vs Small/Large. ETH is the only genuinely different graph.
