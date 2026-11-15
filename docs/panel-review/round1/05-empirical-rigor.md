# Position paper 05: empirical rigour and code audit

Panellist role: sceptical empiricist. I read every Python file, config, sbatch script and the ETL, and checked PyG behaviour against the installed library (torch 2.15, torch_geometric 2.8.1; the project pins PyG for torch 2.2). Scratch checks are in `/tmp/claude-0/-home-user-fin-pse/b1ea4fde-5f89-553f-b4a3-e24c05c55d41/scratchpad/audit/checks.py`.

What I could and could not verify empirically:

- **Run and confirmed:** message-flow direction of the three conv layers, `model.train()` and BatchNorm buffer behaviour, the PR-AUC score bug, parameter counts, checkpoint shape mismatch, `is_edge_attr` slicing of `y`.
- **Verified from library source only:** `NeighborLoader` / `LinkNeighborLoader` sampling direction and seed-edge inclusion. Neither `pyg-lib` nor `torch-sparse` could be installed here, so the loaders would not run.

## Verdict

**CONTINUE WITH MAJOR CHANGES, confidence medium**, and only behind a kill gate.

No number this repository produces today can be trusted. The with-encodings arm cannot even start, and if it did, the "frozen" encoder would be neither frozen nor fed the inputs it was trained on, half its targets would be unobservable, and one of the two reported metrics is computed on the wrong score. All of that is fixable in about two weeks. My larger doubt is scientific: every one of the 12 targets is a closed-form function of a node's incident edges, so I expect a handcrafted-feature baseline to match or beat Fin-PSE. I put the chance that H1 and H2 both survive proper controls at roughly one in four.

## Audit of Part B

| # | Observation | Status | Evidence |
|---|---|---|---|
| 1 | Targets mostly 1-hop; flow-entropy features computed but unused | **PARTLY RIGHT (understated)** | All 12 targets are 1-hop, including `r_2cycle`, which is `min(w, w_rev)` summed over a node's own in/out edges (`etl/aml/transform_window_convert_currencies.py:85-120`). `flow_imbalance` is a deterministic function of two other targets (`:162-164`). `H_out_2_amt` etc. are computed (`:183-228`) and absent from `src/data/ssl_data.py:36-49`. |
| 2 | Out-direction targets unobservable | **CONFIRMED, with one correction** | PyG source: default `sample_direction='forward'` builds the CSC over destinations, so only in-edges of seeds are sampled. Run check: for `ResGatedGraphConv`, `GINEConv` and `PNAConv`, perturbing an out-edge feature leaves the node's output unchanged. `deg_out`, `fan_out`, `vol_out`, both out-gap statistics and `flow_imbalance` (6 of 12) are outside the computational graph. Correction: `r_2cycle` is observable in principle, because a reciprocal edge u→v is an in-edge of in-neighbour v and is sampled at hop 2. |
| 3 | Random node features fixed, 1-d, mismatched downstream | **CONFIRMED, and worse** | `ssl_data.py:78,117` draw noise inside `process()`, so it is cached in the `.pt` file and identical across epochs, runs and seeds. Downstream `x` is all ones (`aml_data.py:145`, `eth_data.py:51`). No config sets `encodings_random_feats`. When set, `mpnn.py:71-72` redraws noise on every forward pass, including evaluation, so test predictions become stochastic and still do not match the fixed-per-node pretraining regime. |
| 4 | Edge-feature shift | **CONFIRMED** | Pretraining: `[Timestamp, Amount USD]`, z-normalised per window (`ssl_data.py:124`). Downstream: `edge_attr[:, :2]` (`mpnn.py:69`), which is `[Timestamp, Amount Received]` in native currency, normalised with train-split statistics over the full period (`aml_data.py:153,253-256`). Test timestamps therefore sit outside any range seen in pretraining. |
| 5 | Encoder BatchNorm not frozen | **CONFIRMED** | `mpnn.py:37` calls `.eval()`; `aml_train.py:92` and `eth_train.py:93` call `model.train()`. Run check: the submodule returns to training mode and `running_mean` updates under `no_grad`; the drifted buffers are saved in `best_state_dict`. It is undone even earlier: `summary()` (`aml_train.py:330`) calls `model.eval()` then `model.train(True)`. |
| 6 | Encoder depth tied to backbone | **CONFIRMED** | `src/networks/fin_pse.py:32`. Additionally the constructor is hard-coded to `FinPSEEncoder(cfg.gnn.dim_in, 2, 64)` (`mpnn.py:27`). |
| 7 | Per-window target z-norm; `-1` fill | **CONFIRMED** | `ssl_data.py:131`; `fill_null(-1)` at `transform_window_convert_currencies.py:79`. |
| 8 | Plain directed backbones | **CONFIRMED** | `mpnn.py:90-146`; all configs have `emlps` false by default (`src/config/gnn.py:11`). |
| 9 | Evaluation protocol | **CONFIRMED, incomplete** | `aml_train.py:237,286-291`; `main.py:59-60`. Only the ETH sbatch scripts pass `--repeat 5`. Test is evaluated each epoch but not used for selection. Missed: no final test number is written anywhere except log lines, and nothing aggregates across seeds. |
| 10 | Same simulator for pretraining and AML downstream | **CONFIRMED** | `configs/ssl/*.yaml:11-12` (HI-Medium) against Small/Large tables. |

Two statements in **Part A** are also wrong. The split is chosen on transaction share only; illicit ratios are computed and never used (`aml_data.py:178-179,193-194`). And `dim_inner` is 20 for PNA on every dataset, not "20 for ETH".

## Additional defects found

### Blocks any valid result

1. **Every `_encodings` config crashes.** All ten set `encodings_file: ''`, so `mpnn.py:30` calls `torch.load("checkpoints/")`. No script runs an encodings config and no file records which checkpoint belongs to which run. Fix: make the path required, assert it exists, log its hash.
2. **Pretraining cannot be launched as shipped.** Both `pretrain.sbatch` files reference `configs/ssl/pretrain.yaml`, which does not exist. The ETL writes to `data/` (`transform_window_convert_currencies.py:266-267`) while `AMLSSL` reads `data/raw/`. The ETL expects `From`/`To` columns that the public CSV does not have (it has two `Account` columns), so an undocumented manual header edit is required. Fix: one script from raw download to checkpoint.
3. **PR-AUC is computed on the wrong score.** `probs = preds[:, 1]` takes the raw class-1 logit (`aml_train.py:55`, `eth_train.py:57`). Ranking by one logit ignores the other. In my synthetic check with 0.2% positives, the same model scored 0.004 by the code's method and 0.277 by `logit1 - logit0`. Fix: use the softmax probability or the logit difference, and report `average_precision_score`.
4. **Seed edges are silently dropped from the loss and from the test metric.** The loader samples at most 100 in-edges per node and does not guarantee the seed edge is in the subgraph; the mask (`aml_train.py:101-104,159-162`) keeps only seed edges that happen to be sampled. An edge into a node with in-degree d is scored with probability about min(1, 100/d). The test set is thus a random, hub-depleted subset that changes with the sampler seed. From source reading; not run. Fix: score `edge_label_index` directly and assert the prediction count equals `len(te_inds)`.
5. **The encoder is run on the truncated mini-batch, not the graph.** `mpnn.py:74-75` applies the encoder to the sampled subgraph. Hop-2 frontier nodes have no in-edges there, so their "encoding" is a function of the sampler, differs between batches, and contains nothing a deeper backbone could not compute from the same subgraph. Fix: precompute encodings once on the full, time-respecting graph and store them as node features.
6. **Half the pretraining objective is noise fitting** (Part B item 2). No validation split and no per-target R² are logged (`ssl_pretrain.py:53-57`), so nobody can tell. Fix: add reverse edges with a direction flag; report held-out R² per target.

### Biases the comparison

7. **Arms differ in capacity and width.** With encodings, the backbone's own embeddings are halved (`mpnn.py:21`) and two projection MLPs are added. Trainable parameters, counted: GIN on AML 36,641 against 49,153 (+34%); GIN on ETH +42%; PNA +11%; plus 58,432 frozen.
8. **The "edge encoding" is a frozen `Linear(2, 64)`** of timestamp and amount (`fin_pse.py:40,47`). It has rank 2 and no structural content, yet takes half the edge channel.
9. **Weight decay is never applied.** `get_optimizer` ignores it for `adam` (`src/util.py:50-51`); every config says `adam` with `weight_decay: 1e-4`. The scheduler is created and never stepped (`main.py:95`).
10. **Targets are heavy-tailed and untransformed.** Volumes and gap variances (seconds squared) are z-scored without a log transform and trained with L1, which is satisfied by predicting the median.
11. **The no-convert and deeper-encoder ablations cannot be loaded** downstream (`mpnn.py:27`; confirmed `RuntimeError` for a 5-feature edge embedding).
12. **No temporal sampling.** A validation or test edge at time t aggregates later edges from the same split. This is conventional for the benchmark but must be stated, and it is at odds with the plan's concern about time leakage.
13. **Currency and payment-format ids are z-scored as continuous values** (`aml_data.py:153,253`).
14. **Model selection:** `best_val_f1` stays 0 after epoch 0, so a worse later epoch can replace it. Selection on argmax F1 at about 0.1% prevalence is high-variance.

### Cosmetic

15. USD conversion pulls live rates from `yfinance` (`etl/aml/utils.py:23-29`): needs network on a compute node and is not reproducible.
16. Processed caches are keyed by table name only (`aml_data.py:43`), so code changes silently reuse stale tensors.
17. `src/networks/mlp.py:11` references a non-existent `cfg.act_dict` (dead code); checkpoint names always say epoch 101; a `NeighborLoader` is rebuilt per window per epoch.

## Strengths

- The chronological split, train-only normalisation statistics and train-only PNA degree histogram are correct.
- Labels never enter the model as inputs. I looked for label leakage through the seed mask and found none; the id bookkeeping is right because train indices are contiguous from zero.
- The checkpoint save and load key structure is consistent.
- The ETL is readable, has row-count assertions in the whole-graph variant, and the windowed design shows real thought about leakage.
- The plan's own evidence list (handcrafted baseline, random encoder, capacity matching) is the right list. The author already knows what would falsify the idea.

## What the current code can and cannot test

| Hypothesis | Testable today? | Why |
|---|---|---|
| H1 | **No** | Encodings arm crashes; if patched, defects 3 to 7 and Part B items 3 to 5 make the comparison uninterpretable. Only the plain baselines run, single seed for AML. |
| H2 | **No** | No handcrafted, random-encoder or capacity-matched arm exists. |
| H3 | **No** | ETH could load an AML checkpoint only because both use two edge features; inputs and scales are unmatched. |
| H4 | **No** | No other encoding is implemented and nothing is timed. Because the encoder runs inside every batch, its cost is per step, the opposite of the precompute-once claim. |

## Minimum fix list before running anything

1. Fix the PR-AUC score; add average precision.
2. Score seed edges directly; assert evaluated count equals split size.
3. Add `--repeat` to all jobs; write one results row per seed to a file.
4. Add reverse edges (or reverse message passing) to encoder and backbone.
5. Log-transform heavy-tailed targets; add a pretraining validation split and per-target R².
6. Use one input convention in both phases: same node features, same edge features, same normalisation.
7. Precompute encodings once per graph, with the encoder in eval mode; override `train()` so it stays there.
8. Give the encoder its own config block (depth, width, edge dimension) saved in the checkpoint.
9. Match trainable parameters and own-embedding width across arms.
10. Apply or delete weight decay; fix epoch-0 selection.
11. Add a smoke test that runs each config for one batch on a synthetic graph.

## Experimental protocol I would require as a reviewer

**Arms**, all at matched trainable parameter count and identical backbone width:

- plain backbone;
- the 12 raw statistics as node features;
- in/out degree only;
- randomly initialised frozen encoder;
- encoder pretrained on shuffled targets;
- Fin-PSE;
- the strongest published adaptation set for this benchmark (reverse message passing, ports, ego ids), with and without Fin-PSE.

**Seeds and tuning.** Ten seeds per arm, covering model initialisation and the sampler. Hyperparameters tuned per arm with the same budget, on validation only.

**Metrics.** Average precision is primary. F1 is secondary, at a threshold chosen on validation, never argmax. Report on the full test split.

**Statistics.** Paired comparison across seeds (Wilcoxon signed-rank), bootstrap confidence intervals over test edges, Holm correction across datasets.

**Convincing pattern.** Fin-PSE beats both the raw-statistics arm and the random-encoder arm with intervals excluding zero on at least three of four AML datasets, and the gain persists on top of the strong adapted backbone. A gain only over the plain directed baseline would not convince me, because that baseline cannot see out-edges and any degree feature repairs it.

## The single experiment that would most change my mind

On HI-Small and LI-Small, with the fixes above, run three arms at ten seeds: the 12 handcrafted statistics computed on the time-respecting graph, frozen Fin-PSE, and a random frozen encoder. If Fin-PSE beats the handcrafted arm by a clear paired margin in average precision, I move to "continue as planned". If it ties or loses, the pretraining contributes nothing over a group-by, and the project should pivot.

## Alternative research directions

### 1. Benchmark-first: a leakage-audited evaluation suite for transaction-graph crime detection

- **Core idea.** Publish a corrected protocol for the synthetic AML benchmark and the Ethereum phishing graph: full-split scoring, temporal neighbour sampling, calibrated metrics, strong tabular and feature baselines.
- **Why promising.** The defects found here (dropped seed edges, logit-as-score, look-ahead neighbours) are inherited from common practice, so published numbers are likely not comparable. A careful audit is low-risk and widely useful.
- **Main risk.** Seen as incremental; needs breadth of baselines to land.
- **First experiment.** Re-evaluate one plain and one adapted GNN with sampled-subset scoring against full-split scoring and report the gap.

### 2. Multi-hop, direction-aware flow targets that cannot be written as a group-by

- **Core idea.** Keep the pretrain-then-freeze design but predict quantities that need message passing: k-step flow entropy (already computed in the ETL), time-respecting reachability counts, directed cycle counts of length 3 to 6, pass-through ratio along temporal paths.
- **Why promising.** It answers the handcrafted-baseline objection directly and targets known laundering motifs such as cycles and scatter-gather.
- **Main risk.** Exact targets are expensive on large graphs; a 2-layer encoder cannot express them.
- **First experiment.** Pretrain on directed 3- and 4-cycle counts with reverse message passing; test whether a linear probe on the encoding recovers injected laundering patterns better than the 12 statistics.

### 3. Temporal-causal sampling and continuous-time encoders

- **Core idea.** Treat the data as an event stream; each prediction uses only earlier edges, with learned time encodings of inter-event gaps instead of a z-scored absolute timestamp.
- **Why promising.** It matches deployment, removes look-ahead, and removes the timestamp extrapolation problem at test time.
- **Main risk.** Scores will drop relative to transductive numbers, which makes comparison with prior work awkward.
- **First experiment.** Rerun the plain baseline with temporal neighbour sampling and relative-time edge features; measure the change in average precision.

### 4. Cross-simulator and synthetic-to-real transfer as the main question

- **Core idea.** Study what transfers between generators and to a real chain, using domain-invariant inputs (rank-normalised amounts, relative times) and no dataset-specific normalisation.
- **Why promising.** It is the only part of the current plan where a learned encoder could beat feature engineering.
- **Main risk.** The answer may be "nothing transfers".
- **First experiment.** Train a detector on one simulator setting, evaluate zero-shot on another illicit-ratio setting and on Ethereum, with and without invariant inputs.

**My preference** is direction 1, with direction 2 as the method contribution if the gate experiment fails.
