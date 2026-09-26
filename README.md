# Business Entity Resolution

Matching 2.2M deduplicated reference entities (S1) against ~10.3M noisy business
records (S2 + S3).

Naive comparison is `2,206,821 x 10,320,219 = 22,774,876,013,799` pairs (~22.8
trillion), so the pipeline is built around **blocking**: only pairs that share a
key are ever compared. This repository currently implements the infrastructure
and the first blocker (exact normalized name) plus full blocking evaluation.

**Status: blocking, matcher V1 and the submission path are implemented; the dense blocker is implemented but not yet calibrated.** See [Roadmap](#roadmap).

---

## Headline finding (read this before planning the model)

Measured against the real normalized names across all 10.3M records, on a
100,000-entity S1 sample:

| Metric | Value |
|---|---|
| Pair recall (normalized name exact match) | **25.79%** |
| Pair recall gained by also dropping separators (`name_key`) | +1.15% |
| **S1 entities for which ALL true matches are retrieved** | **3.74%** |
| S1 entities with at least one true match retrieved | 60.76% |
| Candidate precision of the resulting pairs | ~7.7% |

Two conclusions that shape the rest of the work:

1. **Exact-name blocking alone cannot produce a competitive score.** It recovers
   about a quarter of true matches and leaves 96% of S1 entities with an
   incomplete candidate set. Under a per-entity macro F0.5, an S1 missing even
   one of its matches has a hard ceiling on its own score. This blocker is a
   foundation, not a solution.
2. **Precision is weak even where it matches.** ~7.7% of exact-name candidates
   are true matches: many distinct businesses share a name. Precision work in the
   matcher is not optional, and `beta = 0.5` makes a false positive cost 4x a
   false negative.

Next blockers (token, character n-gram, then multilingual dense retrieval) are
what move recall. Re-run the blocking evaluation after each one.

### Corrected blocker ceiling

The table above measures **exact-name blocking as shipped**. Phase 0.2-0.5
measured something different: **signal coverage on already-known true pairs** -
whether a signal *would* fire on a pair the ground truth already says is a
match. That is not the same as a blocker being able to *propose* it.

| Measurement | True pairs | Kind |
|---|---|---|
| Exact name (`name_norm`) as a blocker | 25.79% | retrievable today |
| Union of the **three lexical generators** (exact name + rare token + char 3-gram) | **82.19%** | generator ceiling |
| Union of all **four** signals (the above + address Jaccard) | 94.99% | **not** a blocker ceiling |

**Do not present 94.99% as the achievable blocker recall.** Address is a
**filter, not a candidate generator**: an address rule is applied to pairs some
other signal already proposed, so it can only remove volume, never create a
candidate. Three lexical generators can therefore reach at most **82.19%** of
true pairs, and no matcher recovers a pair the blocker never proposed. Of the
5.01% irreducible residue (382,722 pairs), 134,718 are cross-script
(transliteration) - the case a character-level signal cannot see by
construction.

The provisional configuration shipped today reaches **57.0123%** measured pair
recall (336,056,756 candidates, J=0.3) - the 82.19% is a signal ceiling, not a
reachable operating point at an acceptable candidate volume. See
[the blocker registry](#the-blocker-registry-provisional-configuration).

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── .gitignore                      # excludes the dataset and generated artifacts
├── configs/
│   └── config.yaml                 # all paths, thresholds and switches
├── notebooks/
│   └── 1.ipynb                     # exploration ONLY - no production logic
├── src/                            # importable, tested, stable code
│   ├── data_loader.py              # config, streaming TSV IO, ground truth
│   ├── normalization.py            # Unicode-safe multilingual normalization
│   ├── blocking.py                 # inverted index + blocker union
│   ├── evaluation.py               # blocking metrics + per-entity F0.5
│   ├── utils.py                    # logging, progress, hashing, id codec, device
│   ├── submission.py               # matching_results.tsv: write, validate, score
│   ├── features.py                 # stub - features live in scripts/extract_pair_features.py
│   └── matching_model.py           # V1 LightGBM matcher + threshold sweep
├── scripts/                        # CLI entry points for heavy work
│   ├── prepare_data.py             # stage 1
│   ├── build_indexes.py            # stage 2 (every blocker enabled in config)
│   ├── generate_candidates.py      # stage 3
│   ├── evaluate_blocking.py        # stage 4
│   ├── extract_pair_features.py    # stage 4b: pair features
│   ├── train_model.py              # stage 5: V1 matcher
│   ├── predict.py                  # stage 6: matching_results.tsv
│   ├── score_submission.py         # validate / score a submission
│   └── fetch_dense_model.py        # one-time bge-m3 download (login node)
├── outputs/                        # generated (gitignored)
└── logs/                           # run logs (gitignored)
```

**Architecture rule:** `notebooks/1.ipynb` is for exploration only. Anything
stable lives in `src/`; anything heavy is runnable from `scripts/`.

> `scripts/evaluate_blocking.py` and `src/utils.py` are additions to the file list
> in the brief. Evaluation had to be command-line runnable and belonged in
> neither `train_model.py` nor the library modules; `utils.py` holds the shared
> primitives (logging, progress, id codec, device detection).

---

## Environment setup

The dataset is **not** in this repository. It lives on the HPC.

### Local / HPC (CPU is enough - a GPU is an optional accelerator)

```bash
git clone <repo-url> entity-resolution
cd entity-resolution

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Only `numpy`, `pandas` and `PyYAML` are required to run stages 1-4, all on CPU.
`tqdm`/`psutil` add nicer progress and RSS logging. The GPU and model packages in
`requirements.txt` are commented out and optional: they are consumed only by the
accelerator-beneficial stages, which resolve a device automatically and fall back
to CPU when those packages are absent. See
[Compute architecture](#compute-architecture).

Verify the install:

```bash
python -m src.normalization        # multilingual normalization self-check
python -c "from src.utils import describe_device; print(describe_device())"
```

---

## Configuration

All paths live in `configs/config.yaml`. **No path is hardcoded in Python.**
Defaults work for a fresh clone with data in `./train` and `./test`.

Override without editing the file - environment variables take precedence over
`config.yaml`:

```bash
export ER_DATA_ROOT=/scratch/challenge/data/train
export ER_TEST_DATA_ROOT=/scratch/challenge/data/test
export ER_WORK_DIR=/scratch/$USER/er_outputs
```

or per command:

```bash
python scripts/prepare_data.py --data-root /scratch/challenge/data/train --work-dir /scratch/$USER/er_outputs
```

Precedence: CLI flag > environment variable > `config.yaml` > built-in default.

Every script also accepts `--config /path/to/other.yaml`.

---

## Compute architecture

**CPU-first, with automatic GPU acceleration for GPU-beneficial stages.**

Every stage runs to completion on CPU, and CPU is the guaranteed fallback. Stages
that benefit from an accelerator obtain one through `utils.resolve_device()` /
`utils.resolve_device_from_config()` — never a hardcoded `"cuda"` — so the same
code takes the GPU when one is present and falls back to CPU otherwise.
`compute.device` in `config.yaml` pins the choice when you need it
(`auto` | `cpu` | `cuda` | `cuda:N` | `mps`).

| Stage | Compute | Why |
|---|---|---|
| Normalization | **CPU** | Unicode + regex string work. A GPU port would put the Indic combining-mark guarantee at risk for no meaningful gain |
| Exact index build / lookup | **CPU** | `argsort` + `searchsorted` over a few million ints |
| Candidate union / dedupe | **CPU** | Chunked per S1, so per-chunk volume is small; GPU transfer overhead would dominate |
| Token / char n-gram blocking | **CPU** | Hash and posting arithmetic |
| Lexical pair features | **CPU** (multiprocess) | `rapidfuzz` is C++ and parallelizes across cores; no GPU edit-distance path worth using |
| Phase 0.2-0.5 blocking statistics | **CPU** (multiprocess) | Per-pair `set`/`Counter` work over 7.6M pairs. Measured: encoding the trigrams a GPU kernel would need costs ~10 µs/string against a 5.4 µs whole-pair CPU budget, and vectorized NumPy came in at 0.1x the plain Python loop - so the accelerator starts behind before it does any work |
| GBDT training | **CPU by default** | For ~20 features the GPU histogram path often loses to a well-threaded CPU build — benchmark `device=cuda` before enabling |
| **Embedding generation** | **GPU when available** | ~12.6M texts, one-time, embarrassingly parallel |
| **Dense retrieval / FAISS** | **GPU when available** | Exact search at this scale is GPU-friendly; needs the index in VRAM (fp16 for 16GB cards) |
| **Batched embedding similarity** | **GPU when available** | Gather + matmul over candidate pairs |
| **Transformer / cross-encoder rerank** | **GPU when available** | Runs only on a small "uncertain" candidate band, so cost stays bounded |

The CPU rows are measurement-driven decisions, not limitations. None of those
stages should grow a GPU path without a benchmark showing it actually wins.
`scripts/benchmark_phase0.py` exists to hold that claim to account: it times the
reference implementation, a flat-batched NumPy arm and a CUDA arm on real pairs,
**checks every arm against the reference before reporting its time**, and sweeps
worker and chunk sizes. Run it before changing the compute policy.

### Workers and memory

`compute.num_workers: 0` means auto: the **physical** core count, clamped to the
number of chunks. Physical rather than logical because this work is python
string/token bound, so hyperthread siblings mostly add contention. There is no
low ceiling - a wide node is used.

The pair pass feeds a process pool from a bounded sliding window, and the queued
chunks hold their pair strings as python objects. `compute.payload_budget_bytes`
(`null` = a quarter of currently-available RAM) caps that queue: when it does not
fit, the chunk size is reduced first and the window second, and the adjustment is
logged. A too-large default batch slows a run down; it never kills one.

Report what the current machine resolves to:

```bash
python -c "from src.utils import describe_device; print(describe_device())"
```

On a Slurm cluster, request a GPU only for the stages marked GPU above; the rest
are CPU jobs. See [HPC notes](#hpc-notes).

---

## Running the pipeline

### Local smoke test (~30 seconds, <400 MB)

```bash
python scripts/prepare_data.py      --splits train --limit 100000 --overwrite
python scripts/build_indexes.py     --limit 100000 --overwrite
python scripts/generate_candidates.py --limit-s1 100000
python scripts/evaluate_blocking.py --split val --no-save
```

Smoke-test recall looks near zero **by design**: it indexes only the first 100k
target rows, so most true matches are not in the index. Use it to check the
plumbing, not the quality.

### Full run

```bash
# 0. Once, on a node with internet (only if the dense blocker is enabled)
python scripts/fetch_dense_model.py --output /scratch/$USER/models/bge-m3 --verify

# 1. Normalize all sources, train and test (~10 min, streaming, ~150 MB RSS)
python scripts/prepare_data.py

# 2. Build the indexes of every blocker enabled in config, for both splits
python scripts/build_indexes.py --split train
python scripts/build_indexes.py --split test

# 3. Candidate pairs -> outputs/candidates/{train,test}_candidate_pairs.tsv
python scripts/generate_candidates.py --split train --workers 0
python scripts/generate_candidates.py --split test  --workers 0

# 4. Blocking evaluation against ground truth (train split)
python scripts/evaluate_blocking.py --split val

# 5. Features: every labelled train entity, and EVERY test entity
python scripts/extract_pair_features.py --split train --sample-fraction 1.0 --workers 32
python scripts/extract_pair_features.py --split test  --sample-fraction 1.0 --workers 32

# 6. Matcher (entity-grouped out-of-fold LightGBM, threshold tuned on val)
python scripts/train_model.py

# 7. Submission (+ a scored dry run on train)
python scripts/predict.py --split test     # -> outputs/submission/matching_results.tsv
python scripts/predict.py --split train    # dry run, macro F0.5 vs ground truth
```

Step 2 builds one index per **enabled** blocker per target source - the same set
step 3 loads. Step 3 verifies the char candidates, so give it the node's cores
with `--workers 0` (auto) or an explicit count. `--blockers exact_name` forces a
single blocker for a quick run without rebuilding anything.

**Artifacts are never silently reused across runs of a different shape.** Each
prepared table carries a provenance sidecar (`*.meta.json`: row limit,
normalization, TSV dialect, raw file size/mtime) and each index a build record
(row limit, blocker cell, key field, prepared-table signature). A stage that finds
a mismatching artifact rebuilds it; `generate_candidates.py` refuses an index that
no longer matches the config. So the smoke test below can be followed by the full
run without `--overwrite`.

**Strict TSV.** Every read and write uses `quoting=QUOTE_NONE`: a `"` in a
business name is an ordinary character (pandas' default would let an unmatched
quote swallow the following rows). `prepare_data.py` also compares parsed rows
against the raw file's physical line count and fails on a mismatch
(`--allow-row-loss` to accept it).

Stage-by-stage reference:

| Command | Reads | Writes | Peak RAM |
|---|---|---|---|
| `prepare_data.py` | `train_source{1,2,3}.tsv` | `outputs/prepared/train_source{1,2,3}_norm.tsv` | ~150 MB |
| `build_indexes.py` | prepared S2/S3 | `outputs/indexes/train_source{2,3}_<blocker>/` (one per enabled blocker) | ~250 MB-1 GB/index |
| `generate_candidates.py` | prepared S1 + indexes | `outputs/candidates/{split}_candidate_pairs.tsv` (`token_df`/`char_jaccard`/`dense_cosine` columns appear when those blockers are enabled; a legacy `candidate_pairs.tsv` is still read for train) | ~300 MB |
| `evaluate_blocking.py` | candidates + ground truth | `outputs/candidates/blocking_metrics_*.json` | ~600 MB |

`prepare_data.py` also writes `{split}_source1_norm.tsv` with a `split` column
(`train`/`val`) per S1 entity.

### Phase 0.2-0.5: blocking statistics

Answers what the signals can and cannot reach before any blocking is built
(signal coverage, residue, candidate census, zero-match analysis). It changes no
analytical definition and produces no predictions.

```bash
# Measure first: arms are verified against the reference, then timed.
python scripts/benchmark_phase0.py --config configs/config.yaml --sample 500000

# The analysis itself. --workers 0 = auto (see "Workers and memory").
python scripts/analyze_blocking_statistics.py --config configs/config.yaml --workers 0
```

Two flags worth knowing on a long run:

* `--timings` records a per-phase wall-clock breakdown in `meta.phase_seconds`,
  which is how you find out *which* phase a slow run is actually spending time in.
* `--resume` reuses the completed per-pair phase from `_ckpt/` under
  `--output-dir`. The checkpoint is keyed on every input that changes the numbers
  (pair count, chunk size, sources, split, prepared corpus), so a stale one is
  recomputed rather than silently trusted; `meta.resumed_phases` records what was
  actually reused.

### Phase 1 Step 0: char-3-gram blocker calibration

Phase 0 measured **signal coverage on known true pairs** - whether a signal
*would* fire on a pair the ground truth already says is a match. It cannot say
whether that signal is *retrievable* as a blocker, or what it costs to retrieve
it. Step 0 measures the other half: the actual **char-3-gram blocker
retrievability x candidate-volume curve** on the real corpus.

This is an **experimental measurement stage**, not a production blocker.
`char_ngram` stays disabled in `config.yaml`, the production blocker
architecture is unchanged, and **no DF cap, rarest-K or Jaccard threshold has
been selected yet** - those are chosen only after inspecting the HPC
calibration results.

```bash
# Full grid: recall x volume for every (DF cap, rarest-K, Jaccard) cell.
python scripts/calibrate_char_blocker.py \
    --config configs/config.yaml \
    --workers 0 \
    --resume \
    --timings

# Volume only: price the whole grid without expanding candidates. Run this
# first - it tells you whether the full grid is affordable before you spend a
# night on it.
python scripts/calibrate_char_blocker.py \
    --config configs/config.yaml \
    --workers 0 \
    --volume-only \
    --timings
```

**Calibration dimensions:** trigram document-frequency cap; rarest-K trigrams
per entity; exact char-3-gram Jaccard verification threshold.

**Retrieval design.** Target-corpus trigram DF -> retain eligible trigrams ->
select the rarest K per entity -> inverted postings -> S1 retrieval -> exact
Jaccard verification -> evaluation against ground truth. Retrieval is an
inverted index; candidates are never produced by comparing every S1 name to
every target name. Verification is bit-identical to Phase 0.1's
`_trigram_jaccard`, so the curve is measured with the analytical definition
already in use, not a substitute.

**Metrics reported per calibration cell.** Blocking pair recall; macro entity
recall; S1 full and partial recall; candidate count; average / median / p90 /
p99 / max candidates per S1; zero-candidate S1 count and fraction; candidate
precision; reduction ratio; and the per-source S2 / S3 breakdown.

**Outputs** (under `<work_dir>/calibration/`):

| File | Contents |
|---|---|
| `char_blocker_calibration.json` | every cell, all metrics, both volume kinds |
| `char_blocker_calibration.csv` | the grid, one row per (set, DF cap, K, threshold) |
| `char_blocker_volume.csv` | candidate volume per cell, bound vs exact |
| `char_blocker_top_trigrams.csv` | trigram DF evidence per source |
| `char_blocker_calibration.md` | the readable summary |
| `_artifacts/` | resumable intermediate stores and indexes |

`_artifacts/` is resumable intermediate data, not a result: it is regenerated
from the prepared corpus and **must not be committed**.

**What Step 0 already settles.**

* `exact_name` is **redundant with char-3-gram for recall** - an exact
  normalized name is contained in char matching, so it adds no pair that char
  blocking cannot already reach. The `char_plus_exact` set equals `char` at every
  threshold.
* **Token blocking is disabled** and will be evaluated separately, as a
  generator.
* **Address is a matcher feature and filter**, not a generator.
* **Embeddings / dense retrieval are deferred** to a later phase.

**Validation status** - local synthetic fixtures only; the full corpus runs on
HPC:

| Suite | Result |
|---|---|
| `tests/test_char_blocker_calibration.py` | 36/36 |
| `tests/test_blocking_statistics.py` | 39/39 |
| `tests/test_compute_utils.py` | 23/23 |
| `tests/test_recall_at_k.py` | 5/5 |

### Step 3: matcher feature extraction (de-risk experiment)

`scripts/extract_pair_features.py` materializes the first matcher feature set on a
**sample of validation candidate pairs**, to size the full run before committing
to it. It is a measurement script, not a pipeline stage: it trains nothing,
changes no blocker, and writes only inside its own experiment directory
(`<candidates_dir>/../experiments/step3_features`, or `--output-dir`), so nothing
it produces can be mistaken for a pipeline artifact.

```bash
python scripts/extract_pair_features.py \
    --config configs/config.yaml \
    --split train \
    --sample-fraction 0.03 \
    --output-dir outputs/experiments/step3_features
```

* **Sampling is by whole S1 entity**, decided by a pure function of the entity id
  (the same `assign_splits` the evaluator uses, plus an independent second bucket
  of the same hash). Sampling whole entities is what keeps the run leak-free: a
  matcher trained on part of an entity's candidate list is still trained on that
  entity's name, address and competitor set.
* **Blank evidence is not zero.** `token_df` is blank on every pair the token
  blocker did not propose and `char_jaccard` on every pair the char blocker did
  not propose; both become `NaN`, never `0`. Every run reports a per-feature
  missingness rate so the blanks stay visible.
* **A failed text join keeps its row.** One row per candidate pair is the
  contract. When an id is missing from the prepared corpus, the text-derived
  features are blanked and `text_join_ok` is 0, while provenance/evidence/the S1
  candidate count - which come from the candidate file, not the join - are kept.
* **No ground truth is read.** There is no label column and no truth file in the
  input path; the split comes from `assign_splits`, the same pure function
  `src/evaluation.py` uses.

#### Parallel feature extraction (`--workers N`)

Phase 1 (scan and sample) stays in the parent whatever `--workers` is, so the
selected entities - and therefore the whole-entity sampling invariant - cannot
depend on the worker count. Only phase 2 is parallelized:

```bash
# same sample, featurized by 8 processes (10-CPU allocation)
python scripts/extract_pair_features.py \
    --config configs/config.yaml --split train \
    --sample-fraction 0.03 --workers 8 \
    --output-dir outputs/experiments/step3_features_w8
```

* `--workers 1` is the default and is the **original single-process path,
  unchanged** - same code path, no shard files, no worker processes.
* The parent partitions the *selected S1 entities* into N contiguous, row-balanced
  groups. Row balance, not entity-count balance: entities differ by an order of
  magnitude in candidate count, so splitting the entity list evenly would leave one
  worker with most of the work. Entity → worker is a pure function of the entity's
  position in the scan and its row count, so the partition is reproducible.
* Each worker loads **only the text its own shard can join to** (its S1 ids plus the
  target ids its rows reference). A filter cannot change a join result - an id is
  kept when it is needed *and* present, and an id absent from the prepared file is a
  join failure with or without the filter - so W workers hold roughly one copy of the
  prepared text between them rather than W copies. The report's
  `parallel.total_lookup_size` is what the node needs at that worker count; compare
  it with `memory.prepared_lookup_estimate` from a `--workers 1` run of the same
  command.
* The merge is in **worker index order, never completion order**, so two runs of the
  same command produce the same bytes whatever order the workers finish in.
  `--workers 2` output is byte-identical to `--workers 1` on the same sample (asserted
  on a synthetic fixture).
* `--workers` is validated against the CPUs this process may actually use (the
  affinity mask, i.e. the scheduler allocation - not the node's core count), and a
  count outside `1..that` is rejected rather than clamped. Nothing hardcodes 48.
* Workers run under `spawn` on every platform: `fork` would inherit the parent's
  pages, which buys nothing here (the parent holds no prepared text) and would
  contaminate the per-worker RSS figures this experiment exists to report.
* Worker shards, per-worker logs and per-worker feature files land in
  `<output-dir>/workers/` - never in the repository root - and are **kept** after a
  successful run, because they are what makes a worker disagreement traceable.
  `--cleanup-shards` removes that directory once the merge has succeeded.
* The end of the log reports workers, selected S1 entities, candidate pairs
  processed, feature rows, elapsed time, peak RSS and throughput, plus per-worker
  rows/seconds/lookup size and the node total. The 48-worker projection is reported
  only in a block explicitly labelled **theoretical/unmeasured**; the measured
  projection is labelled with the worker count it was actually measured at.

Outputs, all inside one experiment directory: `sample_candidates.tsv`,
`features.tsv`, `feature_missingness.csv`, `step3_features_report.json`,
`extract_pair_features.log`, and - for `--workers N > 1` - `workers/`. The report
carries the sample size, scan/shard/merge/feature throughput, peak RSS (per process
and the node total, when parallel), per-feature dtype/min/max/missingness, join and
duplicate counters, output sizes, and a full-scale extrapolation **labelled with a
confidence level** - the projections assume the feature kernel scales near-linearly
with workers and that the sample's names are as long as the real ones, and both
assumptions are stated in the report rather than implied.

`rapidfuzz` is required for the three ratio features; without it they degrade to
NaN and `integrity.rapidfuzz_available` records the fact instead of the run
failing silently.

Validated locally by `tests/test_pair_features.py` (**51/51**, synthetic fixtures
only), which pins the hand-computed value of every similarity, both blank-evidence
rules, the whole-entity sampling invariant (including across a chunk boundary),
duplicate counting, byte-identical output across chunk sizes, and the parallel layer:
the sample is identical at `--workers 1` and `--workers 2`, the partition is
complete/disjoint/deterministic/row-balanced, worker output carries the
single-process schema, the merged `--workers 2` output equals `--workers 1` row for
row, empty partitions and invalid worker counts are handled, and `--workers 1`
creates no worker directory at all. The full 336M-pair run has **not** been executed
anywhere yet.

### HPC notes

* Everything is a plain CLI command - wrap it in your scheduler's batch script.
  Slurm directives are not included because the cluster's configuration was not
  specified; add `#SBATCH` lines appropriate to your site.
* Long stages log progress at a fixed interval and stream their output, so
  `tail -f logs/prepare_data.log` works.
* Set `ER_WORK_DIR` to scratch: `outputs/prepared/` is ~1.5 GB and
  `candidate_pairs.tsv` grows with candidate volume.
* Lower `io.chunksize` in `config.yaml` if a node has little RAM; peak memory
  tracks the chunk size, not the dataset size.
* `--overwrite` is off by default, so re-running a completed stage is a no-op
  (it verifies the existing artifact and skips).
* Request a GPU node only for the stages marked GPU in
  [Compute architecture](#compute-architecture); the rest are CPU jobs. Set
  `compute.device: cpu` to force CPU on a mixed cluster, or `cuda` to fail loudly
  when no GPU was granted - `auto` silently falls back to CPU, which is safe but
  slow for the embedding stages.

---

## How blocking works

`normalized_name -> entity ids`, built per target source.

**Exact-name index.** The textbook implementation is `dict[str, list[int]]`, which
would hold ~4.0M string keys for S2 and cost ~0.7-1 GB before postings. Instead
the index is flat numpy:

| Array | dtype | Size (S2) | Purpose |
|---|---|---|---|
| `key_hashes` | uint64, sorted | ~32 MB | `searchsorted` lookup |
| `key_offsets` | int64 | ~32 MB | slice into `keys_blob` |
| `keys_blob` | concatenated UTF-8 | ~100 MB | exact verification |
| `postings` | int64 | ~40 MB | entity codes, grouped by key |
| `postings_offsets` | int64 | ~32 MB | slice into `postings` |

Queries are answered with `np.searchsorted`; **every hash hit is verified against
the stored string**, so the index cannot emit a spurious pair. A 64-bit hash
collision would cause a missed pair (~4e-7 probability at 4M keys) - blocking
fails toward "miss", never toward "wrong candidate". Keys are hashed with
blake2b, not python's `hash()`, because the latter is salted per process and
would make a persisted index unreadable in the next run.

**Union.** Multiple blockers each return postings for the same S1; the candidate
set is their union. Pairs are packed as `s1_position * 10**11 + entity_code`,
which lets a single `np.unique` do union + dedupe + sort at once. Each pair keeps
a `blockers` provenance column, so you can later see which blocker actually
earns its keep.

The union is never an intersection: a pair proposed by *any* enabled blocker is a
candidate, because blocking's job is a safe over-approximation. The matcher can
reject a false candidate; nothing downstream can recover a true pair that
blocking never proposed.

### The blocker registry (provisional configuration)

Three blockers are implemented, in `src/blocking.py`. Each queries **its own**
normalized column - a token key and a trigram key are not comparable values, so
there is no single shared query column:

| Blocker | Key column | Decision | Per-pair evidence |
|---|---|---|---|
| `exact_name` | `name_norm` | exact key equality | - |
| `token` | `name_norm` | shares one eligible token (boolean, no verification) | `token_df` |
| `char_ngram` | `name_key` | shares a rare trigram, then trigram Jaccard >= `jaccard` | `char_jaccard` |

The **provisional production configuration** is the cell that was measured
end to end - pair recall 57.0123%, 336,056,756 candidates:

```text
exact(name_norm)  UNION  token(name_norm, df<=1000, rarest 1)
                           UNION  char(name_key, df<=1000, rarest 5, J>=0.3)
```

`df_cap` and `rarest_k` are **not tuning knobs**: they define which blocker you
are running. Each blocker keeps, per entity, only its `rarest_k` *eligible* keys,
where eligible means `0 < df <= df_cap` over the **target corpus** (S2 and S3 are
counted separately, and never from S1). Eligibility is applied *before* ranking -
an index-absent key scores `df == 0`, so ranking first would let an unretrievable
key consume one of the entity's K slots. `scripts/build_indexes.py` builds at the
cell directly; that is exactly equivalent to building one loose index and
filtering it down, because eligibility is a prefix-preserving filter over the
`(df, code)` order.

Verification for `char_ngram` is the same `_trigram_jaccard` reference the
calibration used, and a test pins it to
`scripts/analyze_name_differences._trigram_jaccard`. `compute.num_workers` (or
`--workers`) only shards that verification: results are identical at any worker
count.

### The dense blocker (implemented, not yet calibrated)

`dense` (`DenseIndex` in `src/blocking.py`) embeds `name_norm` with **BAAI/bge-m3**
(MIT licence, ~568M parameters, 1024-dim, multilingual) and searches a FAISS
inner-product index: each S1 keeps its `top_k` nearest targets with cosine
`>= min_score`, and the cosine is carried as `dense_cosine` evidence (and as the
`dense_cosine` / `blocker_dense` matcher features). It exists for the pairs no
lexical signal can see - the 134,718 cross-script true pairs above.

* **Offline by construction.** `local_files_only: true` is the default; the model
  is fetched once with `scripts/fetch_dense_model.py` and loaded from that
  directory. Compute nodes never contact the hub.
* **Disabled by default** until the HPC recall x volume run picks `top_k` /
  `min_score`. Enable with `blocking.dense.enabled: true` (plus
  `model_name_or_path`), or pass `--blockers exact_name,token,char_ngram,dense`.
* **Scale.** Encoding ~12.6M names is the GPU stage (`compute.device`); target
  embeddings are stored once as float16. `faiss_factory: Flat` is exact; at 5M x
  1024 prefer an IVF/HNSW factory once its recall cost is measured.
* **Measured on the synthetic smoke test** (`tests/fixtures/synthetic_smoke_test.py`,
  real bge-m3): true cross-script pairs score **0.67-0.84**, overlapping Latin
  near-miss negatives (e.g. "Acme Holdings" vs "Acme Plumbing" at 0.72). Two
  consequences: `min_score` must stay low (default 0.60 - retrieval is
  recall-first), and **no single cosine cut separates them**, so dense-only pairs
  must be decided by the LightGBM matcher, which sees `dense_cosine` next to the
  lexical features - not by the one-feature `--model threshold` baseline.

A **DF=5000 / rarest-K=1 token variant remains an independent background
experiment**, run through `scripts/calibrate_token_blocker.py`. It is
deliberately *not* wired into `configs/config.yaml`, and the measured evidence
does not currently favour it.

**Candidate cap.** `blocking.max_candidates_per_source` (or `--max-candidates`)
limits candidates per S1. This is a recall/precision trade-off, not a detail:
capping silently deletes true matches before the matcher sees them. Choose it
from the `recall_at_k_file_order` numbers in the evaluation report.

---

## Normalization

Multilingual and Unicode-safe, in `src/normalization.py`. The design constraint
that matters:

> **Never drop combining marks.**

India-sourced records appear in Devanagari, Kannada and Bengali. A naive
"strip accents" routine (NFD, then remove all `Mn` characters) is correct for
French and destroys Devanagari - vowel signs and the virama are combining marks,
so `कंस्ट्रक्शंस` would collapse into noise and every Indian business name would
start colliding with every other one.

Pipeline: `NFKC -> per-character table -> case -> whitespace collapse -> truncate`

Accent folding is applied **only to Latin-script characters**, one character at a
time. The table is built once per process over the whole code space and applied
with `str.translate` (C speed). Unassigned code points are excluded, keeping the
table at ~12k entries instead of ~900k.

```
"Orelee's Barbershop"                  -> "orelee s barbershop"
"Café Béque"                           -> "cafe beque"
"B+ Retail Inc"                        -> "b retail inc"
"राम मार्केटिंग प्राइवेट लिमिटेड"          -> "राम मार्केटिंग प्राइवेट लिमिटेड"   (marks preserved)
"ಶಿವಶಕ್ತಿ ವಿದ್ಯಾಲಯ"                       -> "ಶಿವಶಕ್ತಿ ವಿದ್ಯಾಲಯ"                  (marks preserved)
```

Original columns are never overwritten - `business_name` stays exactly as it came
in, because character-level features need the raw text later. `name_key` is an
extra separator-free variant for equality-only blocking.

---

## Validation protocol

**Split by S1 entity, never by pair.** All candidate pairs of one S1 stay in the
same split. Splitting pairs would leak: the same S1 (often with near-identical
address text) would appear in both train and val.

The split is a pure function of the S1 id (blake2b hash -> train/val), so the
candidate generator, the preparer and the evaluator agree on the held-out set
with no shared state to drift. Default 20% val, recorded in the `split` column of
`train_source1_norm.tsv`.

Blocking evaluation reports recall, candidate volume **and** the F0.5 you would
get if every candidate were accepted - that last number is the ceiling the
current blockers impose on any downstream matcher, and it says how much precision
work is left.

**Open question - zero-match entities.** 123,247 S1 entities (5.6%) have an empty
ground-truth list. The challenge's macro-average presumably includes them, which
means predicting *anything* for such an entity scores 0. Both policies are
computed and reported (`f05_accept_all_macro` averaging over entities that have
matches, and `f05_accept_all_macro_score_zero`); which one the leaderboard uses
should be confirmed. `evaluation.zero_match_policy` in `config.yaml` selects the
labelled primary.

`recall_at_k_file_order` is reported for K in 10/25/50/100/200. For an unranked
blocker "first K" is file order, i.e. arbitrary - it bounds what a future
re-ranker can achieve and makes the cost of capping volume explicit. It becomes
meaningful once blockers emit scores.

---

## Data facts (measured, not assumed)

| Fact | Value |
|---|---|
| S1 / S2 / S3 rows | 2,206,821 / 5,034,616 / 5,285,603 |
| Ground-truth rows | 2,206,821 (one per S1, no duplicates) |
| S1 with zero matches | 123,247 (5.6%) |
| Total true matches | 7,638,365 (S2: 3,693,619 / S3: 3,944,746) |
| Matches per S1 | min 0, max 11 (mode 3) |
| S1 with >= 1 S2 / S3 match | 1,919,076 / 1,940,545 |
| S2 / S3 missing `business_address` | 168,967 / 175,916 |
| Missing `business_name` | 0 |
| Entity ids | unique per source, numeric part 2-9 digits, **no leading zeros** |
| Unique `name_norm` (S1 / S2 / S3) | 1,520,684 / 3,949,779 / 4,191,008 |

Two consequences:

* **Address cannot be the primary blocking signal** - ~3.4% of target records have
  none, and ~170k records would be unreachable.
* **Ids pack into int64 losslessly** (no leading zeros), which is what lets the
  ground truth live in ~61 MB instead of ~400 MB of id strings and lets the
  evaluator do set membership with `searchsorted` instead of python sets.

Note: this normalization yields 3,949,779 unique S2 names, while an earlier
figure of ~4,028,180 was reported (~2% more). The difference is explained by the
`punctuation_to_space` and case settings here collapsing slightly more variants
(e.g. `heassociates.com` -> `heassociates com`, `Pvt.` -> `pvt`). The setting is
configurable; if the earlier normalization was validated against something
specific, it can be reproduced by flipping `normalization.punctuation_to_space`.

---

## Outputs

```
outputs/
├── prepared/
│   ├── train_source1_norm.tsv        entity_id, business_name, business_address,
│   ├── train_source2_norm.tsv        country, name_norm, name_key, address_norm,
│   ├── train_source3_norm.tsv        country_norm [, split]
│   └── prepare_manifest.json         row counts + settings (provenance)
├── indexes/
│   ├── train_source2_exact_name/     flat .npy arrays + keys.bin + meta.json
│   ├── train_source2_token/          + the token vocabulary and df table
│   ├── train_source2_char_ngram/     + target name_key blob, for verification
│   ├── train_source3_*/
│   └── index_summary.json
└── candidates/
    ├── {split}_candidate_pairs.tsv   source1_entity_id, matched_entity_id,
    │                                 source, blockers [, token_df]
    │                                 [, char_jaccard]
    ├── {split}_candidate_pairs_stats.json
    └── blocking_metrics_*.json
```

The evidence columns (`token_df`, `char_jaccard`) appear only when their blocker is
enabled, so an exact-name-only run keeps the original four-column schema. A blank
field means the blocker that owns that column did not propose the pair - an
exact-name pair has no char Jaccard. The columns are descriptive only; `blockers`
is what records which generator produced the pair.

`candidate_pairs.tsv` is an **intermediate** artifact, not a submission. Its
format is defined by this project; matched ids are always valid S2/S3 ids and
deduplicated per S1.

`matching_results.tsv` (the graded artifact) is written by `scripts/predict.py`
in the training ground truth's shape - `source1_entity_id<TAB>matched_entity_ids`,
ids comma-separated:

* **every S1 entity exactly once, in S1 file order** - the entity list comes from
  the raw S1 file, so entities the blockers proposed nothing for (absent from every
  candidate and feature file) are still written;
* **a singleton is an exact empty string** - the line is `S1-12<TAB>`, never `nan`;
* ids deduplicated and sorted, LF line endings, no BOM, so reruns are byte-identical.

The file is then validated byte by byte (`src/submission.py`), and `predict.py`
refuses a feature file from a sampled or partial extraction, which would silently
turn the unsampled entities into singletons. `scripts/score_submission.py`
validates any submission and, for the train split, scores it (macro F0.5,
`score_zero`).

---

## Roadmap

**Milestone 1 (done): infrastructure + exact-name blocker + evaluation**

* config-driven paths, streaming IO, compact ground truth
* Unicode-safe multilingual normalization
* flat numpy inverted index with a persisted on-disk format
* union of blockers with provenance
* blocking evaluation: recall, volume, reduction ratio, per-entity F0.5 ceiling
* S1-level validation split

**Milestone 2 (in progress): more blockers, then a matcher**

Order matters - each step should be measured before the next:

1. ~~**Token / inverted-index blocking**~~ - **done.** Implemented as
   `TokenIndex` with the calibrated `(df_cap, rarest_k)` semantics and enabled in
   the provisional configuration. See
   [the blocker registry](#the-blocker-registry-provisional-configuration).
2. ~~**Character n-gram retrieval**~~ - **done.** Implemented as `CharNgramIndex`
   with the calibrated trigram Jaccard verification and enabled in the provisional
   configuration. Step 0 (`scripts/calibrate_char_blocker.py`) measured its
   recall x volume curve before the operating parameters were chosen; see
   [Phase 1 Step 0](#phase-1-step-0-char-3-gram-blocker-calibration). Up to
   82.19% of true pairs is the ceiling for the lexical generators.
3. **TF-IDF / BM25 lexical retrieval**, memory-efficient.
4. **Dense multilingual embedding retrieval** - GPU when available, embeddings
   computed once per record (not per pair) and cached.
5. **Matcher** - threshold on one lexical score first, then gradient-boosted
   trees on lexical + address features, then semantic features. See
   `src/features.py` and `src/matching_model.py` for the planned contract.
6. **Cross-encoder re-ranking** on top candidates only, if it still pays off.

The next step is **step 5, the matcher**, not further blocking: at the measured
operating point the marginal precision of the remaining name-blocking headroom is
far below what the matcher can add by rejecting false candidates.

Since then: the V1 matcher (`src/matching_model.py`, `scripts/train_model.py`),
the submission path (`scripts/predict.py`, `src/submission.py`) and the dense
blocker (step 4, pending calibration) are implemented. `src/features.py` remains a
stub; the feature definitions live in `scripts/extract_pair_features.py`.

### Verification

```bash
python -m pytest tests/                       # unit + integration suites, no downloads
python tests/fixtures/synthetic_smoke_test.py --model-path /path/to/bge-m3
```

The smoke test builds a tiny train/test world (cross-script Devanagari/Kannada
pairs, singletons, an empty-name entity, quote characters) and runs every stage as
a CLI subprocess with the hub switched off. Under pytest it runs when
`ER_DENSE_MODEL` points at a local model, and is skipped otherwise.

---

## Constraints

* **No external data or internet augmentation.** Nothing in this repository
  performs a network lookup; embeddings must come from a locally-run model.
* **Model license and parameter-count constraints** apply to the final solution.
  Record the chosen encoder and its size in `src/features.py` before shipping.
* **CPU-first, GPU-accelerated where it pays.** No stage requires a GPU and no
  stage hardcodes a device: `utils.resolve_device()` returns `cuda` when torch
  and a GPU are present, and `cpu` otherwise. See
  [Compute architecture](#compute-architecture) for the per-stage split.
* **Never materialize the 22.8T cross product.** All stages are chunked and
  streamed; peak memory is governed by `io.chunksize`.

---

## Assumptions made

Listed explicitly since they were not specified in the brief:

1. **Repository root = the challenge folder.** The existing `train/`, `notebooks/`
   and `src/` sat directly in the working directory, so the project root is that
   directory rather than a nested `entity-resolution/`.
2. **Data at `./train` and `./test` by default**, matching the files already
   present. Override with `ER_DATA_ROOT` / `--data-root` for the HPC.
3. **`notebooks/1.ipynb` and `src/normalization.py` were both empty (0 bytes)**, so
   there was no prior normalization logic to reuse despite the brief referring to
   one. Normalization was implemented from scratch and validated against
   Devanagari and Kannada records.
4. **The ground truth covers all 2.2M S1 entities**, including 123,247 with an
   empty match list.
5. **`train_source1.tsv` row order is the canonical S1 order**; candidates are
   emitted in that order.
6. **No scheduler or GPU specifics were assumed** - no `#SBATCH` directives, no
   GPU model names, no HPC paths.
7. **`matching_results.tsv` format is unknown** and deliberately not guessed.
