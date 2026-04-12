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

Examples:

- raw mastery points may be less useful than categorical versatility flags
- solo-lane setup or "primer" indicators may matter more when paired with jungler gank tendencies than when treated in isolation
- matchup-sensitive role weights are worth exploring only if they can be represented cleanly and measured against baseline

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
