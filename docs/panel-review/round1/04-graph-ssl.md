# Position paper 04: graph self-supervised learning and foundation models

Basis: I read all of `src/`, `etl/aml/transform_window_convert_currencies.py`, `etl/aml/utils.py` and the configs. Nothing was executed (no PyTorch in the sandbox), so claims about PyG sampling come from reading, not running. Web search worked but arxiv.org was unreachable, so papers are cited from search snippets, or marked "(from memory)".

## Verdict

**PIVOT.** Confidence: high that the current objective and protocol should not be run as planned; medium on which replacement wins.

The pretext task regresses quantities a group-by computes exactly in one pass, so pretraining distils a cheap exact feature extractor into an expensive approximate one, and half of the targets are not functions of the encoder's input. The frozen-concatenation protocol then hands the backbone a fixed function of information it already has, so a positive H1 would be uninterpretable. The underlying question (can an encoder pretrained on transaction graphs transfer?) is open and worth keeping, but it needs a different objective, input parameterisation and evaluation.

## Strengths

- **Open niche.** Search found self-supervision for AML only as single-dataset link prediction or contrastive learning (LaundroGraph, arXiv 2210.14360), and multi-network pretraining only for ERC-20 token networks (MiNT, arXiv 2406.10426). I found no established pretrained encoder for directed, temporal, amount-attributed multigraphs.
- **Structure-only inputs.** Dropping account identity is the right choice for cross-graph transfer, as in GPSE (arXiv 2307.07107) and GFSE (arXiv 2504.10917).
- **The windowed ETL is reusable.** It turns one ledger into many graphs and already computes 2-step flow entropies (`etl/aml/transform_window_convert_currencies.py:183-228`) that the target list ignores.
- **The plan names its own falsifier.** Evidence item 2 (feed the 12 statistics directly) is the correct test.
- **The pivot is cheap.** The codebase is small and registry-based.

## Weaknesses (most severe first)

1. **The objective is distillation, not self-supervision.** Eleven of the 12 targets (`src/data/ssl_data.py:36-49`) are one-hop aggregates produced by a polars group-by (`etl/...convert_currencies.py:128-176`). GPSE is worth pretraining because its targets (Laplacian eigenvectors, random-walk, heat-kernel, cycle encodings) are expensive, global and beyond what the downstream MPNN can compute. Here the downstream PNA (mean/min/max/std with degree scalers, `src/networks/mpnn.py:111-112`) computes in-degree, in-volume and time spread exactly in its first layer. The representation is forced to encode only what is already free. **H2 is false by construction for these targets; H4 holds only against the wrong comparator** (the honest one is the group-by).

2. **Half the targets are unobservable, and the loss rewards a constant.** Part B item 2 is correct. `NeighborLoader` (`src/train/ssl_pretrain.py:23`) samples in-edges of seeds, `ResGatedGraphConv` passes messages source to target (`src/networks/fin_pse.py:33`), and there are no reverse edges. Out-degree, fan-out, out-volume and both outgoing gap statistics are outside the seed's computational graph; flow imbalance is half-observable; reciprocity is visible only through the 2-hop path via a reciprocal partner. Under L1 the optimum for an unobservable target is its conditional median. Targets and amounts are z-normalised without a log transform (`ssl_data.py:124,131`), so heavy-tailed volume and degree have a median near a constant. Fan-in needs neighbour identity, which exists in pretraining only as fixed 1-d noise (`ssl_data.py:117`) and vanishes downstream where features are all ones (`src/data/aml_data.py:145`, `src/data/eth_data.py:51`). Gap statistics need order statistics that a sum aggregator cannot form. What survives is roughly in-degree and in-volume. **H1 loses its mechanism.**

3. **Frozen concatenation adds no information and confounds three things.** The encoder sees a subset of the backbone's own input (`mpnn.py:69`) through the same direction-blind two-hop receptive field. Unlike GPSE (random features, deep network, virtual node; from memory) it cannot inject anything the backbone lacks. Any gain must come from (a) two projection MLPs (`mpnn.py:39-50`), (b) four message-passing layers instead of two, or (c) a fixed nonlinear basis. Meanwhile the native embedding is halved (`mpnn.py:21`). Random-init frozen encoders are known strong baselines (the Random-Init row in Deep Graph Infomax; from memory), and none of the controls exists in code. **H1 cannot support H2.**

4. **The encoder is not validated, not frozen, and fed shifted inputs.** There is no held-out pretext evaluation: every node is a training node (`ssl_data.py:142-144`) and the loop never evaluates. Part B item 5 is correct: `self.encodings.eval()` (`mpnn.py:37`) is undone by `model.train()` (`src/train/aml_train.py:92`, `src/train/eth_train.py:93`), so BatchNorm statistics adapt to downstream batches. That partly hides the shift in Part B items 3 and 4 (both confirmed: per-window USD z-scores versus whole-period native-currency z-scores). The protocol is "frozen weights, adapted normalisation", which no ablation isolates. Minor: with `optimizer: adam` weight decay is silently dropped (`src/util.py:51`), contradicting the brief.

5. **The transfer study has one target.** HI-Medium to Small/Large is the same simulator, so it is resampling, not transfer. ETH is a single target, and its inputs are an absolute timestamp and raw amount, neither comparable across ledgers. MiNT's evidence is that transfer scales with the number of pretraining networks; this project has one. **H3 is untestable as designed.**

6. **The regime is wrong for pretraining.** AMLworld is fully labelled. Pretraining pays under label scarcity, and tuned controls often erase its gains (Hu et al., ICLR 2020; Sun et al., NeurIPS 2022; both from memory). The plan has no label-efficiency curve.

## Improvements (by value per unit of effort)

1. **Pretext audit (hours).** Hold out 20% of windows and report per-target R². Repeat on HI-Small through the downstream input path. I predict out-direction R² near zero.
2. **H2 controls (a day).** Same backbone with (i) the exact 12 statistics as node features, (ii) the encoder randomly initialised and frozen, (iii) the encoder trained end to end. Override `train()` so the encoder stays in eval mode, and decouple its depth from `cfg.gnn.layers_mp` (`fin_pse.py:32`).
3. **Unit-free inputs and a bidirectional encoder (days).** Replace the absolute timestamp with log inter-event time to the endpoint's previous event; replace amount with log1p followed by a per-graph quantile; add reverse edges with a direction flag and sample both directions. This is a precondition for any cross-ledger claim.
4. **Replace the objective (a week).** Use masked edge-attribute modelling: mask 30% of edges' amount bucket, time-gap bucket and payment format, and predict them by cross-entropy from bidirectional context (GraphMAE-style; from memory). Add a causal head that predicts each node's next-window out-behaviour from the current window. Neither target is a deterministic function of the input. Keep statistic regression only for expensive targets: temporal 3- to 5-cycle counts, scatter-gather counts, multi-hop pass-through.
5. **Evaluate as an encoder.** Linear probe on frozen embeddings, then frozen versus adapter versus full fine-tuning, at 1%, 10% and 100% of training labels, 5 seeds, PR-AUC.
6. **A real transfer matrix.** Leave-one-source-out over AMLworld HI/LI, AMLSim, Ethereum phishing, ERC-20 networks, Elliptic++ and DGraph-Fin (the last two from memory; check they carry the needed attributes). Report scratch, random-frozen, in-domain and out-of-domain pretraining, including negative transfer.

## The single experiment that would most change my mind

After fixing BatchNorm freezing and input mismatch, run five arms on HI-Small, LI-Small and ETH with GIN and PNA, 5 seeds, at 1%, 10% and 100% of training labels: (A) backbone alone; (B) backbone plus the exact 12 statistics; (C) backbone plus random-init frozen encoder; (D) backbone plus pretrained frozen encoder; (E) backbone plus the same encoder trained end to end.

- **Towards CONTINUE:** D beats the best of B, C and E by more than one seed standard deviation in PR-AUC at any label budget, on at least two datasets including ETH.
- **Confirms PIVOT:** D is within noise of C, or B matches or beats D. I predict B ≥ D ≈ C at every budget, with E ≥ D at 100%.

## Alternative research directions

1. **Multi-ledger masked-event pretraining (my favourite).** *Idea:* improvements 3 and 4 trained jointly on many ledgers, evaluated leave-one-ledger-out with linear probes and few-shot labels. *Why:* MiNT shows transfer grows with the number of transaction networks, and sequence models pretrained on card transactions transfer to fraud (Skalski et al., arXiv 2401.01641); nobody has combined graph context with unit-free tokenisation for AML. *Risk:* simulators dominate the corpus and the model learns simulator artefacts. *First experiment:* pretrain on HI-Medium, LI-Medium and 20 ERC-20 networks; linear-probe HI-Small and ETH at 1% labels against scratch and random-frozen.

2. **In-context relational or tabular foundation models as the baseline to beat.** *Idea:* cast laundering detection as a temporal prediction task over account and transaction tables and apply a pretrained in-context model with no training. *Why:* KumoRFM-2 (arXiv 2604.12596) reports few-shot in-context results surpassing supervised models on RelBench (vendor-reported). If that holds here, bespoke GNN pretraining is moot. *Risk:* a 0.1% positive rate and cyclic patterns that local context sampling misses. *First experiment:* a tabular in-context model on edge features plus both endpoints' 12 statistics, a few thousand context rows, against XGBoost and GIN on HI-Small.

3. **Amortised temporal-motif encoder (GPSE done properly for finance).** *Idea:* regress temporal cycle, scatter-gather and bipartite-pattern counts with an encoder using reverse message passing, port numbering and ego IDs (Multi-GNN, arXiv 2306.11586). *Why:* these features drive strong tree-model AML results (Graph Feature Preprocessor; from memory) and are costly to mine, so amortisation has real value and an expressivity theory behind it. *Risk:* sampled MPNNs cannot count long cycles. *First experiment:* held-out R² above 0.8 for 3- and 4-cycle participation on HI-Medium, then zero-shot R² on LI-Small and ETH.

4. **Simulator-prior pretraining and sim-to-real.** *Idea:* generate graphs from randomised simulator parameters with planted typologies, train an in-context detector, test on real ledgers. *Why:* every AMLworld result is sim-to-sim; generalist anomaly detectors (AnomalyGFM, arXiv 2502.09254) show zero-shot cross-graph detection is feasible but assume node attributes. *Risk:* the prior misses real typologies. *First experiment:* train on AMLworld, test zero-shot on AMLSim and Ethereum phishing, and report the gap.
