#!/usr/bin/env python
"""End-to-end synthetic smoke test: real CLIs, real bge-m3, no network, tiny data.

Generates a miniature S1/S2/S3 train + test dataset, then runs the whole pipeline
exactly as it runs on the cluster - every stage as its own CLI subprocess, with the
Hugging Face hub switched off (``HF_HUB_OFFLINE=1``) to prove there is no runtime
network dependency:

    prepare_data -> build_indexes (train, test) -> generate_candidates (train, test)
    -> extract_pair_features (train, test) -> train_model -> predict (test, train)

and asserts the properties the audit fixes exist for:

* **C4 dense blocker** - a Devanagari S1 name and its romanized target (and a
  Kannada one) share no token and no trigram, so no lexical blocker can propose
  them; the dense blocker must, and the pair must reach the submission.
* **C1 singletons** - an S1 with a unique name and an S1 with an *empty* name (no
  candidates at all, so absent from every candidate and feature file) must both be
  written as ``S1-x<TAB>`` - an exact empty string, never ``nan`` or a missing row.
* **C2** every test S1 with candidates is featurized; **C3** train and test candidate
  files coexist; **H3** ``build_indexes`` (no ``--blockers``) builds every enabled
  blocker, dense included; **H4** a literal ``"`` in a name survives.

Usage (the model directory comes from ``scripts/fetch_dense_model.py``)::

    python tests/fixtures/synthetic_smoke_test.py --model-path D:/AMLC/models/bge-m3
    ER_DENSE_MODEL=/scratch/$USER/models/bge-m3 python tests/fixtures/synthetic_smoke_test.py

Under pytest it runs only when ``ER_DENSE_MODEL`` points at a local model, and is
skipped otherwise (the download-free dense tests live in tests/test_dense_blocker.py).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

RAW_HEADER = ["entity_id", "business_name", "business_address", "country"]

# ---------------------------------------------------------------------------
# The synthetic world
# ---------------------------------------------------------------------------
TRAIN_S1 = [
    ("S1-101", "Acme Holdings LLC", "12 Market Street, Springfield", "US"),
    ("S1-102", "Blue Ocean Seafood", "4 Harbor Road, Portland", "US"),
    ("S1-103", 'Joe\'s "Famous" Pizza', "88 Elm Ave, Chicago", "US"),
    ("S1-104", "राम मार्केटिंग प्राइवेट लिमिटेड", "सदर बाजार, दिल्ली", "IN"),
    ("S1-105", "श्री गणेश ट्रेडर्स", "एमजी रोड, पुणे", "IN"),
    ("S1-106", "ಶಿವಶಕ್ತಿ ವಿದ್ಯಾಲಯ", "ಜಯನಗರ, ಬೆಂಗಳೂರು", "IN"),
    ("S1-107", "लक्ष्मी ज्वेलर्स", "चांदनी चौक, दिल्ली", "IN"),
    ("S1-108", "Zyxwvut Quantum Aerospace Consortium", "1 Nowhere Plaza", "US"),
    ("S1-109", "Greenfield Organic Farms", "Route 9, Albany", "US"),
    ("S1-110", "Sunrise Dental Clinic", "22 Lake Road, Austin", "US"),
    ("S1-111", "Patel Brothers Grocery", "Devon Ave, Chicago", "US"),
    ("S1-112", "Northwind Logistics Ltd", "Dock 5, Seattle", "US"),
    ("S1-113", "कुमार मेडिकल स्टोर", "गांधी नगर, जयपुर", "IN"),
    ("S1-114", "Silverline Auto Repair", "9 Garage Lane, Denver", "US"),
]
TRAIN_S2 = [
    ("S2-201", "ACME Holdings, LLC", "12 Market St, Springfield", "US"),
    ("S2-202", "Acme Plumbing Services", "40 Pipe Rd, Springfield", "US"),
    ("S2-203", "Blue Ocean Sea Food", "4 Harbor Rd, Portland", "US"),
    ("S2-204", "Joes Famous Pizza", "88 Elm Avenue, Chicago", "US"),
    ("S2-205", "Ram Marketing Private Limited", "Sadar Bazar, Delhi", "IN"),
    ("S2-206", "Ram Electricals", "Karol Bagh, Delhi", "IN"),
    ("S2-207", "Shivashakti Vidyalaya", "Jayanagar, Bengaluru", "IN"),
    ("S2-208", "Greenfield Organic Farm", "Route 9, Albany NY", "US"),
    ("S2-209", "Sunrise Dental", "22 Lake Rd, Austin", "US"),
    ("S2-210", "Kumar Medical Store", "Gandhi Nagar, Jaipur", "IN"),
    ("S2-211", "Silverline Auto Repairs", "9 Garage Ln, Denver", "US"),
    ("S2-212", "Harbor Freight Tools", "1 Tool Way, Portland", "US"),
    ("S2-213", "Lakshmi Sweets", "Chandni Chowk, Delhi", "IN"),
]
TRAIN_S3 = [
    ("S3-301", "acme holdings llc springfield", "", "US"),
    ("S3-302", "Shree Ganesh Traders", "MG Road, Pune", "IN"),
    ("S3-303", "Lakshmi Jewellers", "Chandni Chowk, Delhi", "IN"),
    ("S3-304", "Patel Bros Grocery", "Devon Avenue, Chicago", "US"),
    ("S3-305", "Ganesh Hardware", "Camp, Pune", "IN"),
    ("S3-306", "Quantum Computing Institute", "5 Lab Road, Boston", "US"),
    ("S3-307", "Northstar Logistics", "Pier 3, Tacoma", "US"),
]
TRAIN_GT = [
    ("S1-101", "S2-201,S3-301"),
    ("S1-102", "S2-203"),
    ("S1-103", "S2-204"),
    ("S1-104", "S2-205"),
    ("S1-105", "S3-302"),
    ("S1-106", "S2-207"),
    ("S1-107", "S3-303"),
    ("S1-108", ""),
    ("S1-109", "S2-208"),
    ("S1-110", "S2-209"),
    ("S1-111", "S3-304"),
    ("S1-112", ""),
    ("S1-113", "S2-210"),
    ("S1-114", "S2-211"),
]

TEST_S1 = [
    ("S1-501", "Riverside Bakery & Cafe", "3 River Rd, Hartford", "US"),
    ("S1-502", "गुप्ता इलेक्ट्रॉनिक्स", "लाजपत नगर, दिल्ली", "IN"),
    ("S1-503", "ಕೃಷ್ಣ ಹೋಟೆಲ್", "ಮೈಸೂರು", "IN"),
    ("S1-504", "Xylophonic Nebula Biotech Syndicate", "77 Orbit Blvd", "US"),
    ("S1-505", "", "Unknown", "US"),
    ("S1-506", 'Mama Rosa\'s "Original" Trattoria', "14 Via Roma, Newark", "US"),
]
TEST_S2 = [
    ("S2-601", "Riverside Bakery and Cafe", "3 River Road, Hartford", "US"),
    ("S2-602", "Gupta Electronics", "Lajpat Nagar, Delhi", "IN"),
    ("S2-603", "Gupta Hardware Mart", "Chawri Bazar, Delhi", "IN"),
    ("S2-604", "Riverside Auto Body", "90 River Rd, Hartford", "US"),
    ("S2-605", "Nebula Coffee House", "2 Star St, Denver", "US"),
]
TEST_S3 = [
    ("S3-701", "Krishna Hotel", "Mysuru", "IN"),
    ("S3-702", "Mama Rosas Original Trattoria", "14 Via Roma, Newark NJ", "US"),
    ("S3-703", "Krishna Sweets", "Mysuru", "IN"),
    ("S3-704", "Pacific Biotech Labs", "8 Harbor Dr, San Diego", "US"),
]

CROSS_SCRIPT_TEST_PAIRS = [("S1-502", "S2-602"), ("S1-503", "S3-701")]
UNIQUE_NAME_SINGLETON = "S1-504"
NO_CANDIDATE_SINGLETON = "S1-505"
QUOTED_S1 = ("S1-506", 'Mama Rosa\'s "Original" Trattoria')


def _write_tsv(path: Path, header: list[str], rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(row) + "\n")


def build_world(root: Path, model_path: Path) -> Path:
    """Write the raw dataset and a config whose every path points into ``root``."""
    _write_tsv(root / "train" / "train_source1.tsv", RAW_HEADER, TRAIN_S1)
    _write_tsv(root / "train" / "train_source2.tsv", RAW_HEADER, TRAIN_S2)
    _write_tsv(root / "train" / "train_source3.tsv", RAW_HEADER, TRAIN_S3)
    _write_tsv(root / "train" / "train_ground_truth.tsv", ["source1_entity_id", "matched_entity_ids"], TRAIN_GT)
    _write_tsv(root / "test" / "test_source1.tsv", RAW_HEADER, TEST_S1)
    _write_tsv(root / "test" / "test_source2.tsv", RAW_HEADER, TEST_S2)
    _write_tsv(root / "test" / "test_source3.tsv", RAW_HEADER, TEST_S3)

    import yaml

    with open(REPO / "configs" / "config.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["paths"].update(
        {
            "data_root": str(root / "train"),
            "test_data_root": str(root / "test"),
            "work_dir": str(root / "work"),
            "prepared_dir": str(root / "work" / "prepared"),
            "index_dir": str(root / "work" / "indexes"),
            "candidates_dir": str(root / "work" / "candidates"),
            "log_dir": str(root / "logs"),
        }
    )
    config["io"]["chunksize"] = 3  # several chunks even at this size
    config["compute"].update({"device": "cpu", "num_workers": 1})
    config["evaluation"]["split"]["val_fraction"] = 0.5  # enough val entities to tune on
    config["evaluation"]["zero_match_policy"] = "score_zero"
    # The production lexical blockers stay as configured; dense is switched on with
    # the local bge-m3. min_score is lower than the production default so the
    # candidate set also carries negatives for the threshold to be tuned against.
    config["blocking"]["dense"].update(
        {
            "enabled": True,
            "model_name_or_path": str(model_path),
            "local_files_only": True,
            "batch_size": 16,
            "top_k": 3,
            "min_score": 0.45,
        }
    )
    path = root / "config.yaml"
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True)
    return path


# ---------------------------------------------------------------------------
# Running the stages
# ---------------------------------------------------------------------------
def run_stage(name: str, argv: list[str], config_path: Path, env: dict, log_dir: Path) -> float:
    command = [sys.executable, str(REPO / "scripts" / f"{name}.py"), "--config", str(config_path), *argv]
    started = time.time()
    completed = subprocess.run(command, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - started
    tag = "_".join([name, *[a.strip("-") for a in argv if not a.startswith("/") and ":" not in a]])[:80]
    (log_dir / f"{tag}.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        tail = "\n".join((completed.stdout + completed.stderr).splitlines()[-40:])
        raise RuntimeError(f"stage {name} {' '.join(argv)} exited {completed.returncode}:\n{tail}")
    print(f"  [ok] {name:<22} {' '.join(argv):<60} {elapsed:6.1f}s", flush=True)
    return elapsed


def read_tsv(path: Path):
    import csv

    import pandas as pd

    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_filter=False, quoting=csv.QUOTE_NONE)


class Checks:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, passed: bool, detail: str = "") -> None:
        self.results.append((name, bool(passed), detail))
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}", flush=True)

    @property
    def ok(self) -> bool:
        return all(passed for _, passed, _ in self.results)


def run(model_path: Path, root: Path) -> Checks:
    config_path = build_world(root, model_path)
    work = root / "work"
    stage_logs = root / "stage_logs"
    stage_logs.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        # No network at runtime: any hub call would now fail instead of silently working.
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    checks = Checks()

    print(f"workspace: {root}\nmodel    : {model_path}\n\nstages:", flush=True)
    run_stage("prepare_data", ["--splits", "train,test"], config_path, env, stage_logs)
    for split in ("train", "test"):
        run_stage("build_indexes", ["--split", split], config_path, env, stage_logs)  # no --blockers (H3)
    for split in ("train", "test"):
        run_stage("generate_candidates", ["--split", split, "--workers", "1"], config_path, env, stage_logs)
    run_stage("extract_pair_features", ["--split", "train", "--sample-fraction", "1.0"], config_path, env, stage_logs)
    run_stage("extract_pair_features", ["--split", "test", "--sample-fraction", "1.0"], config_path, env, stage_logs)
    run_stage(
        "train_model",
        ["--model", "threshold", "--score-feature", "dense_cosine", "--folds", "2"],
        config_path, env, stage_logs,
    )
    run_stage("predict", ["--split", "test"], config_path, env, stage_logs)
    run_stage("predict", ["--split", "train"], config_path, env, stage_logs)
    run_stage("score_submission", [str(work / "submission" / "matching_results.tsv"), "--split", "test"],
              config_path, env, stage_logs)

    print("\nchecks:", flush=True)
    # H3: every enabled blocker was built by the default build_indexes invocation.
    expected = [f"{split}_{source}_{blocker}" for split in ("train", "test")
                for source in ("source2", "source3") for blocker in ("exact_name", "token", "char_ngram", "dense")]
    missing = [name for name in expected if not (work / "indexes" / name / "meta.json").is_file()]
    checks.check("H3 build_indexes default built all 4 enabled blockers x 2 sources x 2 splits",
                 not missing, f"missing {missing}" if missing else f"{len(expected)} indexes")
    dense_meta = json.loads((work / "indexes" / "test_source2_dense" / "meta.json").read_text(encoding="utf-8"))
    checks.check("C4 dense index is real bge-m3 (1024-dim, local path, offline)",
                 dense_meta["dim"] == 1024 and dense_meta["settings"]["local_files_only"] is True,
                 f"dim={dense_meta['dim']} model={dense_meta['settings']['model_name_or_path']}")

    # C3: split-aware candidate files, no overwrite.
    train_candidates = work / "candidates" / "train_candidate_pairs.tsv"
    test_candidates = work / "candidates" / "test_candidate_pairs.tsv"
    train_table, test_table = read_tsv(train_candidates), read_tsv(test_candidates)
    checks.check(
        "C3 train and test candidate files coexist",
        set(train_table["source1_entity_id"]) <= {r[0] for r in TRAIN_S1}
        and set(test_table["source1_entity_id"]) <= {r[0] for r in TEST_S1},
        f"train={len(train_table)} rows, test={len(test_table)} rows",
    )

    # C4: the cross-script pairs are proposed by dense and by nothing else.
    for s1, target in CROSS_SCRIPT_TEST_PAIRS:
        row = test_table[(test_table["source1_entity_id"] == s1) & (test_table["matched_entity_id"] == target)]
        provenance = row["blockers"].iloc[0] if len(row) else "(not proposed)"
        cosine = row["dense_cosine"].iloc[0] if len(row) else ""
        only_dense = bool(len(row)) and all(p.endswith(":dense") for p in provenance.split(","))
        checks.check(f"C4 cross-script {s1}->{target} retrieved by dense only",
                     only_dense, f"blockers={provenance} dense_cosine={cosine}")
    checks.check(f"C1 {NO_CANDIDATE_SINGLETON} (empty name) has no candidates at all",
                 NO_CANDIDATE_SINGLETON not in set(test_table["source1_entity_id"]))

    # C2: every test S1 with candidates was featurized.
    test_features = read_tsv(work / "experiments" / "step3_features_test" / "features.tsv")
    with_candidates = set(test_table["source1_entity_id"])
    featurized = set(test_features["source1_entity_id"])
    checks.check("C2 every test S1 with candidates is featurized", with_candidates == featurized,
                 f"{len(featurized)}/{len(with_candidates)} (the old val-only filter would keep ~50% here)")
    checks.check("C2 test features cover every candidate row", len(test_features) == len(test_table),
                 f"{len(test_features)} feature rows / {len(test_table)} candidate rows")

    # H4: a literal quote survives preparation.
    prepared = read_tsv(work / "prepared" / "test_source1_norm.tsv")
    quoted = prepared.loc[prepared["entity_id"] == QUOTED_S1[0], "business_name"]
    checks.check("H4 literal '\"' in a business name survives strict TSV",
                 len(quoted) == 1 and quoted.iloc[0] == QUOTED_S1[1], repr(quoted.iloc[0]) if len(quoted) else "row lost")

    # C1: the submission.
    submission = work / "submission" / "matching_results.tsv"
    raw = submission.read_bytes()
    lines = raw.decode("utf-8").split("\n")[:-1]
    rows = dict(line.split("\t", 1) for line in lines[1:])
    checks.check("C1 header is exact", lines[0] == "source1_entity_id\tmatched_entity_ids", repr(lines[0]))
    checks.check("C1 one row per test S1, in S1 file order",
                 [line.split("\t", 1)[0] for line in lines[1:]] == [r[0] for r in TEST_S1],
                 f"{len(lines) - 1} rows")
    for s1 in (UNIQUE_NAME_SINGLETON, NO_CANDIDATE_SINGLETON):
        line = f"{s1}\t".encode()
        checks.check(f"C1 singleton {s1} written as exact empty string",
                     (b"\n" + line + b"\n") in raw and rows.get(s1) == "", f"line bytes={line!r}")
    checks.check("C1 no NA spelling anywhere", not any(v.strip() in {"nan", "NaN", "None", "null", "NA"} for v in rows.values()))
    checks.check("exact/fuzzy Latin matches reach the submission",
                 rows.get("S1-501") == "S2-601" and rows.get("S1-506") == "S3-702",
                 f"S1-501 -> {rows.get('S1-501')!r}, S1-506 -> {rows.get('S1-506')!r}")

    # C4 plumbing: nothing downstream of the blocker drops a dense-only pair. Its
    # lexical evidence is blank/zero by construction, so a NaN or join mistake would
    # silently kill exactly these rows.
    cross_cosines = {}
    for s1, target in CROSS_SCRIPT_TEST_PAIRS:
        feature = test_features[(test_features["source1_entity_id"] == s1)
                                & (test_features["matched_entity_id"] == target)]
        candidate = test_table[(test_table["source1_entity_id"] == s1)
                               & (test_table["matched_entity_id"] == target)]
        ok = (len(feature) == 1 and feature["blocker_dense"].iloc[0] == "1"
              and feature["text_join_ok"].iloc[0] == "1"
              and abs(float(feature["dense_cosine"].iloc[0]) - float(candidate["dense_cosine"].iloc[0])) < 1e-3)
        cross_cosines[(s1, target)] = float(candidate["dense_cosine"].iloc[0]) if len(candidate) else float("nan")
        checks.check(f"C4 dense-only row {s1}->{target} reaches the features intact", ok,
                     f"blocker_dense={feature['blocker_dense'].iloc[0] if len(feature) else '-'} "
                     f"dense_cosine={feature['dense_cosine'].iloc[0] if len(feature) else '-'}")
        # V2 context: the cross-script pair is its S1's best dense candidate - the
        # signal that lets a model accept a 0.69 cosine a global cut would reject.
        rank = feature["s1ctx_dense_cosine_rank"].iloc[0] if len(feature) else ""
        gap = feature["s1ctx_dense_cosine_gap_to_best"].iloc[0] if len(feature) else ""
        checks.check(f"V2 {s1}->{target} is its S1's top dense candidate (s1ctx rank 1, gap 0)",
                     rank != "" and float(rank) == 1.0 and float(gap) == 0.0, f"rank={rank} gap={gap}")

    # ...and through scoring and the writer: with the threshold set just below the
    # cross-script cosines (a PLUMBING check - the value is derived from the
    # retrieved cosines, it is not a tuned model), the pairs are written and the
    # singletons still come out as "".
    override = round(min(cross_cosines.values()) - 0.01, 4)
    plumbing_out = work / "submission" / "plumbing_matching_results.tsv"
    run_stage("predict", ["--split", "test", "--threshold", str(override), "--output", str(plumbing_out)],
              config_path, env, stage_logs)
    plumbing_rows = dict(line.split("\t", 1) for line in plumbing_out.read_text(encoding="utf-8").splitlines()[1:])
    for s1, target in CROSS_SCRIPT_TEST_PAIRS:
        checks.check(f"C4 dense-only match {s1}->{target} is written (threshold override {override})",
                     target in plumbing_rows.get(s1, "").split(","), f"{s1} -> {plumbing_rows.get(s1)!r}")
    checks.check(f"C1 singletons stay exact \"\" at threshold {override}",
                 plumbing_rows.get(UNIQUE_NAME_SINGLETON) == "" and plumbing_rows.get(NO_CANDIDATE_SINGLETON) == "",
                 f"{UNIQUE_NAME_SINGLETON} -> {plumbing_rows.get(UNIQUE_NAME_SINGLETON)!r}, "
                 f"{NO_CANDIDATE_SINGLETON} -> {plumbing_rows.get(NO_CANDIDATE_SINGLETON)!r}")

    # V2 conflicts: no target is ever written for two S1 entities.
    for label, table in (("tuned", rows), ("override", plumbing_rows)):
        owners: dict[str, list[str]] = {}
        for s1, joined in table.items():
            for target in filter(None, joined.split(",")):
                owners.setdefault(target, []).append(s1)
        shared = {t: s for t, s in owners.items() if len(s) > 1}
        checks.check(f"V2 no target written for two S1 entities ({label} submission)", not shared,
                     f"shared={shared}" if shared else f"{len(owners)} targets, each with one S1")

    report = json.loads((work / "submission" / "train_matching_results_report.json").read_text(encoding="utf-8"))
    model_meta = json.loads((work / "experiments" / "v1" / "model" / "model_meta.json").read_text(encoding="utf-8"))
    comparison = report["target_conflicts"]["comparison"]
    checks.check("train dry run scored end to end, both conflict modes", "score_all" in report and len(comparison) == 2,
                 f"train macro F0.5 (score_zero) keep-best={comparison['keep-best']['all']:.4f} "
                 f"off={comparison['off']['all']:.4f}; GT targets with >1 S1="
                 f"{report['target_conflicts']['ground_truth']['targets_with_multiple_s1']}")

    # Diagnostic, deliberately NOT a pass/fail check: whether the *tuned* one-feature
    # baseline admits the cross-script pairs is a model-quality property of a handful
    # of val entities, not a smoke property. It is reported because it is the finding
    # that matters for the real run.
    tuned = model_meta["threshold"]
    admitted = [f"{s1}->{t}" for (s1, t) in CROSS_SCRIPT_TEST_PAIRS if t in rows.get(s1, "").split(",")]
    print(f"  [INFO] tuned dense_cosine threshold = {tuned:.4f}; cross-script cosines = "
          f"{', '.join(f'{c:.4f}' for c in cross_cosines.values())}; admitted by the tuned baseline: "
          f"{admitted or 'none'}", flush=True)
    if not admitted:
        print("         A single global cosine cut cannot separate cross-script true pairs from Latin "
              "near-miss negatives (see README). This needs the LightGBM matcher, which sees "
              "dense_cosine together with the lexical features.", flush=True)
    return checks


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", default=os.environ.get("ER_DENSE_MODEL"),
                        help="local bge-m3 directory (scripts/fetch_dense_model.py); default $ER_DENSE_MODEL")
    parser.add_argument("--workdir", default=None, help="workspace (default: a fresh temp dir)")
    parser.add_argument("--keep", action="store_true", help="keep the workspace for inspection")
    args = parser.parse_args(argv)
    if not args.model_path or not Path(args.model_path).is_dir():
        print("a local model directory is required: --model-path <dir> or ER_DENSE_MODEL=<dir>\n"
              "  fetch it once: python scripts/fetch_dense_model.py --output <dir>", file=sys.stderr)
        return 2

    root = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="er_smoke_"))
    started = time.time()
    try:
        checks = run(Path(args.model_path).resolve(), root)
    except RuntimeError as error:
        print(f"\nSMOKE TEST ERROR: {error}", file=sys.stderr)
        print(f"workspace kept for inspection: {root}", file=sys.stderr)
        return 1
    passed = sum(ok for _, ok, _ in checks.results)
    print(f"\n{passed}/{len(checks.results)} checks passed in {time.time() - started:.0f}s")
    if checks.ok and not args.keep:
        shutil.rmtree(root, ignore_errors=True)
    else:
        print(f"workspace: {root}")
    return 0 if checks.ok else 1


def test_synthetic_smoke():
    """pytest entry point: runs only when ER_DENSE_MODEL names a local model."""
    import pytest

    model = os.environ.get("ER_DENSE_MODEL")
    if not model or not Path(model).is_dir():
        pytest.skip("set ER_DENSE_MODEL to a local bge-m3 directory to run the dense smoke test")
    assert main(["--model-path", model]) == 0


if __name__ == "__main__":
    raise SystemExit(main())
