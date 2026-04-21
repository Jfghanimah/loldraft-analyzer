# LoL Draft Analyzer Roadmap

This document merges the old roadmap and the old TODO into a single plan that matches the current repo reality.

## Project Consensus

The project already has useful legacy data, a live SQLite pipeline, and working PyTorch training code. Because of that, the next phase is not "start collecting data from scratch." The next phase is to improve model accuracy using the data we already have.

Agreed direction:

3. Keep humans in the loop for training runs for now.
4. Document the training environment well enough that local GPU runs are predictable.
5. Prefer raw, meaningful signals that let the model discover useful structure over brittle hand-authored scoring schemes when possible.


## Main Goal

Raise predictive accuracy with the current dataset before investing in more data generation.

That means the near-term work should concentrate on:

- feature refinement
- data-pipeline cleanup
- better baselines
- training stability
- experiment repeatability

## Near-Term Priorities

### 1. Lock Down A Trustworthy Baseline

Before changing too much, we need a stable baseline that can be reproduced.

Tasks:

- run and record a baseline result for the current two-stage pipeline
- document exact train command, model artifact, and validation metric
- make it easy to compare future experiments against that baseline

### 2. Improve Accuracy With Existing Data

This is the highest-priority technical track.

Candidate improvements that fit the current direction:

- preserve strict positional inputs across all 10 slots
- stop relying on blue/red perspective swapping as the long-term augmentation strategy
- add an explicit side-aware feature so the model can learn map asymmetry
- refine lane interaction features beyond the current blue-minus-red lane difference
- revisit masking and regularization choices in pretraining and fine-tuning
- add a simple baseline model for comparison so we know whether transformer complexity is earning its keep
- explicitly test micro-synergy features over champion pairs and triplets to address the near-zero-repeat nature of full 5v5 compositions
- explore dynamic matchup modifiers rather than relying only on static champion-level priors

### 3. Audit Existing Feature Ideas Before Expanding Data Collection

Some feature ideas from the old TODO are still valuable, but they should be staged carefully.

Good candidates to evaluate after the baseline is stable:

- player rank
- KDA
- vision score
- DPS
- healing
- surrender rate
- fast-climber or smurf-style indicators
- mastery and versatility categories such as OTP, Two-TP, and Flex
- recent activity or recency signals
- other trustworthy player-level features already available from current data sources

Constraint:

- only promote features that can be sourced and maintained reliably
- do not block current accuracy work on speculative feature engineering
- be skeptical of raw mastery totals by themselves because of diminishing-return effects

### 3A. Feature Engineering Principles

Meeting consensus added a few useful guardrails for feature work:

- prefer raw or lightly processed signals over static human-assigned champion labels
- if a categorical abstraction is more stable than a raw count, prefer the categorical version
- look for features that help the model generalize across unseen drafts rather than memorize exact team compositions
- avoid hand-authoring champion semantics such as damage type, engage type, peel, scaling, range, or synergy tags as primary model inputs; champion meaning should be learned from outcomes and draft context
- prefer architectural biases that expose learnable interaction paths, such as allied pair interactions, lane matchups, and cross-team pair relations, without manually naming what those interactions mean

Examples:

- raw mastery points may be less useful than categorical versatility flags
- solo-lane setup or "primer" indicators may matter more when paired with jungler gank tendencies than when treated in isolation
- matchup-sensitive role weights are worth exploring only if they can be represented cleanly and measured against baseline

### 3B. Compositional Generalization Over Static Priors

The long-term model should learn why a draft works rather than memorize historical answer keys. A champion-role pair such as Yuumi jungle should be bad because the model composes a learned Yuumi representation with a learned jungle-role representation and discovers that the resulting token lacks clear speed, sustain, gank threat, or frontline value. The same principle should let the model reason about rare but temporarily strong meta shifts rather than blindly rejecting them because their historical frequency is low.

Implications:

- champion identity, role, team side, player history, and draft context should remain separate learnable inputs before attention
- population champion-role win-rate and frequency priors are useful measured baseline features, but they are also potential lazy paths
- do not hand-author champion semantic tags such as damage type, engage class, peel, scaling, or bot-lane archetype as the primary solution
- use ablations to decide when population priors are still helping versus when they are suppressing learned interaction reasoning
- prioritize self-supervised draft pretraining as the cleaner way to improve champion geometry without injecting a brittle cheat sheet

Masked draft pretraining is the preferred next structural experiment: hide one slot from a real draft, train the model to predict the missing champion from the other nine slots plus role/team/global context, then fine-tune the same encoder on win/loss. This should force the embedding space and attention heads to learn which champions fit which roles and compositions before the noisy binary outcome head is attached.

### 4. Training Performance And GPU Utilization

Training has been inconsistent, and the 2080 Ti is not always being used well enough.

Tasks:

- profile dataloader throughput and batch delivery
- inspect whether Windows worker settings are limiting input throughput
- verify batch size, pin-memory behavior, and host-to-device transfer efficiency
- check whether model size or training loop structure is the true bottleneck
- document realistic expectations for when low GPU utilization is unavoidable versus when it indicates a pipeline problem

### 5. Environment Documentation

The project needs operational documentation, not just model ideas.

Write down:

- how local CUDA training is set up
- which PyTorch build is expected
- how to tell whether CUDA is actually active
- what to do when falling back to CPU
- any known version gotchas around NVIDIA drivers, CUDA, and Windows

This should reduce repeated setup confusion and make future experiments easier to reproduce.

## Secondary Priorities

### Compact V2 Data Infrastructure

The raw-JSON SQLite cache does not scale. The current local database has about 663k games and already takes about 56 GiB, mostly because full Riot match JSON averages roughly 82 KB per game. The next data layer should parse each API response once, keep compact typed facts, and discard raw JSON by default.

Plan:

- collect fresh matches into compact Parquet datasets under `ml/save_data/lol_dataset_v1/`
- keep SQLite only for crawler state such as queues, processed PUUIDs, and seen match IDs
- configure storage mode and compact dataset path in `config/scraper.json`
- store compact match rows, participant-history rows, and later model-ready training examples as partitioned Parquet
- discard full raw Riot match JSON after successful compact extraction
- migrate the existing SQLite DB once using structured columns and `participant_history`, not repeated raw-JSON training reads
- support multiple operators by merging compact Parquet shard directories and deduplicating by `match_id`
- process prepared training examples per platform in parallel while preserving chronological order within each platform
- current validation: the compact dataset has been migrated from the local SQLite source and matches ranked solo queue row counts exactly at 662,952 matches and 6,629,520 participants

Current collection workflow:

- live collection command: `python -m ml.data.data_api_sqlite`
- required secret: `RIOT_API_KEY` in `.env` or `.env.local`
- optional operator state overrides: `LOL_DRAFT_DB_PATH` and `LOL_DRAFT_COLLECTOR`
- active config: `config/scraper.json` uses compact storage into `ml/save_data/lol_dataset_v1/`, queue 420, and three seed regions: NA, EUW, and KR
- after collecting more compact facts, rebuild model-ready examples with `python -m ml.data.prepare_training_examples --dataset-dir ml/save_data/lol_dataset_v1 --overwrite`

### Scraper / Training Coexistence

The scraper should be able to run continuously in the background without colliding with training runs. SQLite WAL mode already allows concurrent readers and one writer, but the training pipeline needs to be verified safe to run alongside an active scraper.

Plan:

- verify the training pipeline opens the DB read-only (or at minimum never writes during a training run)
- ensure the scraper uses WAL mode so training reads are not blocked by write commits
- document the expected workflow: scraper runs persistently, training is launched on-demand against the current snapshot

### Side-Aware Data Pipeline

The current dataset augmentation doubles data by swapping team perspectives. That helped bootstrap the model, but it also hides real game asymmetry.

Plan:

- replace swap-based augmentation as the default assumption
- introduce an explicit side feature or another representation that preserves map-side effects
- compare results directly against the old augmentation approach

### Simple Baseline Model

A multi-hot or similarly simple baseline is still useful, not because the project should move backward, but because it creates a sanity check.

Plan:

- add a lightweight baseline model over current draft data
- compare it against the transformer pipeline
- use the result to judge whether architecture changes are actually worthwhile

### Champion Representation

One failure mode from earlier iterations was effectively treating champion IDs like meaningful numeric values when they are just identifiers. The current embedding-based direction is the correct fix and should stay central.

Plan:

- keep champion identity representation embedding-based
- avoid any design that assumes sequential champion IDs encode similarity
- use embeddings, pair features, and interaction features to learn champion neighborhoods and matchup structure

### Better Experiment Logging

Right now the code can train, but the project documentation does not yet describe a disciplined experiment loop.

Plan:

- define what gets recorded for each run
- track model settings, dataset source, validation metric, and artifact path
- make comparisons across runs easy to interpret

### Compact Training Dataset

The unified trainer should stop rebuilding the dense feature matrix from SQLite on every run. The intended direction is a prepared Parquet dataset:

- compact facts are append-only and reusable
- model-ready training examples are built from compact facts in chronological order
- repeated model experiments read the prepared training examples directly
- feature schema versions and manifests make stale caches explicit
- unified training can use prepared Parquet through `--training-data-dir`
- schema v3 adds smoothed champion-role population win-rate and log-normalized champion-role frequency per player slot, giving the model explicit priors for rare off-role or troll picks without preserving raw API JSON

### Current Experiment Baseline

Current compact schema v3 flat model:

- command: `python -m ml.trainer.train --training-data-dir ml/save_data/lol_dataset_v1`
- best epoch: 12/20
- best validation loss: 0.7027
- best validation accuracy: 57.72%
- approximate validation win BCE after subtracting auxiliary loss: 0.6748
- artifact: `ml/save_data/best_unified_win_predictor.pth`

First architecture A/B:

- command: `python -m ml.trainer.train --training-data-dir ml/save_data/lol_dataset_v1 --architecture team_compare --save-path ml/save_data/best_unified_win_predictor_team_compare.pth --train-log-path ml/save_data/latest_train_log_team_compare.txt`
- change: keep all 10 post-transformer slot embeddings, then add explicit blue pool, red pool, team difference, absolute team difference, and lane-by-lane differences before the prediction head
- best validation loss: 0.7025 at epoch 14
- best validation accuracy during the loss-selected run: 57.85% at epoch 13, but that epoch was overwritten by the lower-loss epoch 14 checkpoint
- approximate validation win BCE at best loss: 0.6746
- follow-up command: `python -m ml.trainer.train --training-data-dir ml/save_data/lol_dataset_v1 --architecture team_compare --selection-metric val_acc --save-path ml/save_data/best_unified_win_predictor_team_compare_acc.pth --train-log-path ml/save_data/latest_train_log_team_compare_acc.txt`
- saved accuracy-selected checkpoint validation accuracy: 57.77% by evaluator, with best logged validation accuracy 57.76% at epoch 16
- interpretation: the structural comparison path is mildly positive and does not hand-code champion semantics, but the gain is small; it should be treated as a promising architecture baseline, not proof that the model has solved draft interaction reasoning
- validation note: auxiliary gold and dragon heads are weakly useful, but game length is currently worse than the mean baseline and should be down-weighted, removed, or redesigned in a future A/B

Second architecture A/B:

- command: `python -m ml.trainer.train --training-data-dir ml/save_data/lol_dataset_v1 --architecture pairwise --selection-metric val_acc --save-path ml/save_data/best_unified_win_predictor_pairwise_acc.pth --train-log-path ml/save_data/latest_train_log_pairwise_acc.txt`
- change: add learned allied pair products plus cross-team pair products and differences over post-transformer slot embeddings
- evaluator validation accuracy: 57.75%
- cost: roughly 2x slower per epoch than `team_compare`
- interpretation: neutral/negative. The idea is directionally aligned with learned champion interaction modeling, but this implementation is too expensive for no clear accuracy gain and should not replace the active `team_compare` checkpoint

Auxiliary-loss A/B:

- command: `python -m ml.trainer.train --training-data-dir ml/save_data/lol_dataset_v1 --architecture team_compare --selection-metric val_acc --aux-target-weights 0.25 0.15 0.15 0.0 --save-path ml/save_data/best_unified_win_predictor_team_compare_no_gamelen_acc.pth --train-log-path ml/save_data/latest_train_log_team_compare_no_gamelen_acc.txt`
- change: keep gold and dragon auxiliary targets, but remove game-length loss from the shared objective
- best validation accuracy: 57.75%
- interpretation: neutral/slightly negative versus active `team_compare`. Do not promote, but keep configurable auxiliary weights for future cleaner A/B runs
- reproducibility note: the trainer now accepts `--seed` and seeds PyTorch/CUDA at run start; tiny accuracy differences around 0.1 percentage points should be treated as noise unless reproduced

Current active architecture:

- command: `python -m ml.trainer.train --training-data-dir ml/save_data/lol_dataset_v1 --architecture cls_global --selection-metric val_acc --model-dim 96 --model-heads 4 --model-layers 4 --model-ff 384 --dropout 0.15 --weight-decay 0.05 --label-smoothing 0.15 --seed 42 --save-path ml/save_data/best_unified_win_predictor_cls_global_l4_acc.pth --train-log-path ml/save_data/latest_train_log_cls_global_l4_acc.txt`
- change: replace manual `team_compare` readout with a learned CLS token and a separate global patch/region token; prediction reads only the CLS token after attention
- evaluator validation accuracy: 57.83%
- interpretation: small metric gain, cleaner architecture, and better auxiliary game-length behavior. This is the active checkpoint in `ml/save_data/best_unified_win_predictor.pth`

Population-prior ablation:

- command: `python -m ml.trainer.train --training-data-dir ml/save_data/lol_dataset_v1 --architecture cls_global --drop-population-priors --selection-metric val_acc --model-dim 128 --model-heads 8 --model-layers 4 --model-ff 512 --dropout 0.15 --weight-decay 0.05 --label-smoothing 0.15 --seed 42 --save-path ml/save_data/best_unified_win_predictor_cls_global_nopriors_dim128_l4_acc.pth --train-log-path ml/save_data/latest_train_log_cls_global_nopriors_dim128_l4_acc.txt`
- change: remove only `champ_role_win_rate` and `champ_role_frequency` from dense inputs, reducing per-slot dense features from 16 to 14
- evaluator validation accuracy: 57.46%
- interpretation: negative. The population priors are useful signal in the current dataset/settings and should stay in the active baseline. The ablation switch remains useful for future checks

## Deferred / Backlog

These are real ideas, but they are not the current top priority.

### New Large-Scale Data Expansion

Do this after current-data accuracy improvements plateau or we prove data scarcity is the bottleneck.

### Agent-Run Training

For now, humans should launch real training jobs. A future automation track can explore whether agents should coordinate experiment runs, queue jobs, or manage retraining workflows.

Questions for that future track:

- how much autonomy should agents have before spending GPU time
- how experiment approval should work
- how model artifacts and metrics should be registered safely

### Website / Product Surface

A draft-analysis website may make sense later, but it should not distract from validating model quality first.

## What Not To Do Right Now

- do not treat data collection as the first missing piece
- do not assume the old roadmap phases reflect current implementation status
- do not push agent autonomy ahead of reproducible human-run experiments
- do not add broad feature complexity without a measurable baseline to compare against

## Definition Of Good Progress

Progress over the next stretch should look like this:

1. a documented baseline
2. one or more accuracy-focused experiments using current data
3. clearer understanding of GPU bottlenecks on the 2080 Ti
4. usable documentation for CUDA and local training setup
5. a tighter, more reality-based backlog for future features and automation
