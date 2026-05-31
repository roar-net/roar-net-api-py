#!/usr/bin/env python3
"""
Cross-language validation: ROAR-NET Python vs Julia TSP implementations.

Usage:
  # Run all algorithms on all instances (100 stochastic trials)
  python tests/cross_validate.py

  # Dry run: 1 trial per algorithm, quick check
  python tests/cross_validate.py --dry-run

  # Specify instances and algorithms
  python tests/cross_validate.py --instances test5,eil51 --algorithms greedy,best

  # Adjust budget and trials
  python tests/cross_validate.py --budget 5.0 --trials 50

  # Skip download
  python tests/cross_validate.py --no-download
"""

import argparse
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTANCES_DIR = PROJECT_ROOT / "instances"
JULIA_PROJECT_DIR = PROJECT_ROOT / "julia"
TSPLIB_URL = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"

# Add TSP example to path
sys.path.insert(0, str(PROJECT_ROOT / "examples" / "tsp"))
from tsp import Problem, Solution
import roar_net_api.algorithms as alg
try:
    from scipy.stats import ks_2samp
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Instance definitions
# ---------------------------------------------------------------------------

INSTANCES = {
    "test2":  {"file": "test2.tsp",   "n": 2,   "opt": 2},
    "test3":  {"file": "test3.tsp",   "n": 3,   "opt": 3},
    "test5":  {"file": "test5.tsp",   "n": 5,   "opt": None},
    "test10": {"file": "test10.tsp",  "n": 10,  "opt": None},
    "eil51":  {"file": "eil51.tsp",   "n": 51,  "opt": 426},
    "berlin52": {"file": "berlin52.tsp", "n": 52, "opt": 7542},
    "kroA100": {"file": "kroA100.tsp", "n": 100, "opt": 21282},
    "ch150":   {"file": "ch150.tsp",   "n": 150, "opt": 6528},
    "tsp225":  {"file": "tsp225.tsp",  "n": 225, "opt": 3916},
}

# Algorithms: name -> (deterministic, needs_init)
ALGORITHMS = {
    "greedy": {
        "deterministic": True,
        "py_fn": lambda prob, budget: alg.greedy_construction(prob),
    },
    "best": {
        "deterministic": True,
        "py_fn": lambda prob, budget: alg.best_improvement(prob, alg.greedy_construction(prob)),
    },
    "first": {
        "deterministic": False,
        "py_fn": lambda prob, budget: alg.first_improvement(prob, alg.greedy_construction(prob)),
    },
    "beam": {
        "deterministic": False,
        "py_fn": lambda prob, budget: alg.beam_search(prob, bw=10),
    },
    "grasp": {
        "deterministic": False,
        "py_fn": lambda prob, budget: alg.grasp(prob, budget),
    },
    "rls": {
        "deterministic": False,
        "py_fn": lambda prob, budget: alg.rls(prob, alg.greedy_construction(prob), budget),
    },
    "sa": {
        "deterministic": False,
        "py_fn": lambda prob, budget: alg.sa(prob, alg.greedy_construction(prob), budget, 30.0),
    },
}


# ---------------------------------------------------------------------------
# Instance management
# ---------------------------------------------------------------------------

def ensure_instances(allow_download: bool = True, instance_names: Optional[list] = None):
    """Ensure instance files exist. Download TSPLIB if missing."""
    INSTANCES_DIR.mkdir(parents=True, exist_ok=True)

    if instance_names is None:
        instance_names = list(INSTANCES.keys())

    missing = []
    for name in instance_names:
        info = INSTANCES[name]
        path = INSTANCES_DIR / info["file"]
        if not path.exists():
            if name.startswith("test"):
                print(f"WARNING: Missing test instance: {path}")
                missing.append(name)
            elif allow_download:
                url = TSPLIB_URL + info["file"]
                print(f"Downloading {url}...")
                try:
                    urllib.request.urlretrieve(url, path)
                    print(f"  -> saved to {path}")
                except Exception as e:
                    print(f"  FAILED: {e}")
                    missing.append(name)
            else:
                missing.append(name)

    if missing:
        print(f"\nMissing instances: {', '.join(missing)}")
        print("Re-run with --download or manually place files in instances/")
        return False
    return True


# ---------------------------------------------------------------------------
# Python runners
# ---------------------------------------------------------------------------

def run_python(instance_path: Path, algo_name: str, seed: int, budget: float) -> tuple:
    """Run a Python algorithm in-process. Returns (obj, elapsed)."""
    random.seed(seed)

    with open(instance_path) as f:
        prob = Problem.from_textio(f)

    start = time.perf_counter()
    sol = ALGORITHMS[algo_name]["py_fn"](prob, budget)
    elapsed = time.perf_counter() - start

    obj = sol.objective_value()
    return obj, elapsed


# ---------------------------------------------------------------------------
# Julia runners
# ---------------------------------------------------------------------------

def run_julia_trials(instance_path: Path, algo_name: str, seed0: int, budget: float, n_trials: int = 1) -> list:
    """Run Julia algorithm for n_trials in a single subprocess. Returns list of (obj, elapsed)."""
    benchmark_script = JULIA_PROJECT_DIR / "examples" / "tsp" / "benchmark.jl"

    cmd = [
        "julia",
        "--project=" + str(JULIA_PROJECT_DIR),
        str(benchmark_script),
        algo_name,
        str(seed0),
        str(budget),
        str(n_trials),
    ]

    with open(instance_path) as f:
        instance_data = f.read()

    result = subprocess.run(
        cmd,
        input=instance_data,
        capture_output=True,
        text=True,
        timeout=max(budget, 10) * n_trials + 120,
    )

    if result.returncode != 0:
        print(f"  Julia error (algo={algo_name}): {result.stderr.strip()}")
        return []

    results = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Format: seed OBJ=val TIME=val
        parts = line.split()
        if len(parts) >= 3 and parts[1].startswith("OBJ="):
            obj_str = parts[1].split("=", 1)[1]
            time_str = parts[2].split("=", 1)[1]
            obj = None if obj_str == "none" else float(obj_str)
            elapsed = float(time_str)
            results.append((obj, elapsed))

    return results


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def ks_test(sample1, sample2):
    """Two-sample Kolmogorov-Smirnov test. Returns p-value (approximate)."""
    if not _HAS_SCIPY:
        return None
    _, pval = ks_2samp(sample1, sample2)
    return pval


def describe(values):
    """Return (mean, std, minimum, maximum) for a list of values."""
    if not values:
        return None, None, None, None
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0, min(values), max(values)


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def auto_budget(n: int) -> float:
    """Pick a reasonable wall-time budget based on instance size."""
    if n <= 10:
        return 0.1
    elif n <= 100:
        return 1.0
    else:
        return 5.0


def compare_algorithm(instance_path, instance_name, instance_info, algo_name, algo_info, cli_budget, trials, quiet=False):
    """Run cross-language comparison for one algorithm on one instance."""
    n = instance_info["n"]
    opt = instance_info["opt"]
    det = algo_info["deterministic"]
    budget = cli_budget if cli_budget and cli_budget > 0 else auto_budget(n)

    n_trials = 1 if det else trials

    # ---- Python trials ----
    py_objs = []
    py_times = []
    for t in range(n_trials):
        seed = t + 1
        obj, elapsed = run_python(instance_path, algo_name, seed, budget)
        if obj is not None:
            py_objs.append(obj)
            py_times.append(elapsed)

    # ---- Julia trials (all in one subprocess) ----
    jl_objs = []
    jl_times = []
    jl_results = run_julia_trials(instance_path, algo_name, 1, budget, n_trials)
    for obj, elapsed in jl_results:
        if obj is not None:
            jl_objs.append(obj)
            jl_times.append(elapsed)

    # ---- Results ----
    py_mean, py_std, py_min, py_max = describe(py_objs)
    jl_mean, jl_std, jl_min, jl_max = describe(jl_objs)

    # Gap to known optimum
    py_gap = None
    if opt is not None and py_mean is not None:
        py_gap = (py_mean - opt) / opt * 100
    jl_gap = None
    if opt is not None and jl_mean is not None:
        jl_gap = (jl_mean - opt) / opt * 100

    # Statistical test
    pvalue = None
    if not det and len(py_objs) > 1 and len(jl_objs) > 1:
        pvalue = ks_test(py_objs, jl_objs)

    return {
        "instance": instance_name,
        "n": n,
        "algorithm": algo_name,
        "deterministic": det,
        "opt": opt,
        "py_count": len(py_objs),
        "py_mean": py_mean,
        "py_std": py_std,
        "py_min": py_min,
        "py_max": py_max,
        "py_gap": py_gap,
        "py_time_mean": statistics.mean(py_times) if py_times else None,
        "jl_count": len(jl_objs),
        "jl_mean": jl_mean,
        "jl_std": jl_std,
        "jl_min": jl_min,
        "jl_max": jl_max,
        "jl_gap": jl_gap,
        "jl_time_mean": statistics.mean(jl_times) if jl_times else None,
        "ks_pvalue": pvalue,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def format_val(v, decimals=1):
    if v is None:
        return "-"
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    return f"{v:.{decimals}f}"


def generate_report(results, markdown=True):
    if markdown:
        return generate_markdown_report(results)
    else:
        return generate_text_report(results)


def generate_markdown_report(results):
    lines = []
    lines.append("# Cross-Language Validation Report")
    lines.append(f"\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n## Summary\n")
    lines.append(f"| Instance | n | Algorithm | Lang | Count | Mean | Std | Min | Max | Gap% | Time(s) | KS p-val |")
    lines.append(f"|----------|---|-----------|------|-------|------|-----|-----|-----|------|---------|----------|")

    for r in results:
        inst = r["instance"]
        n = r["n"]
        algo = r["algorithm"]
        opt = r["opt"]
        det = r["deterministic"]

        for lang in ["py", "jl"]:
            count = r[f"{lang}_count"]
            mean = r[f"{lang}_mean"]
            std = r[f"{lang}_std"]
            mn = r[f"{lang}_min"]
            mx = r[f"{lang}_max"]
            gap = r[f"{lang}_gap"]
            tmean = r[f"{lang}_time_mean"]
            ks = r["ks_pvalue"] if lang == "jl" else None

            opt_str = f" (opt={opt})" if opt is not None else ""
            gap_str = format_val(gap, 1) if gap is not None else "-"
            ks_str = f"{ks:.4f}" if ks is not None else "-"

            row = (
                f"| {inst}{opt_str} | {n} | {algo} | {lang} "
                f"| {count} | {format_val(mean)} | {format_val(std)} "
                f"| {format_val(mn)} | {format_val(mx)} "
                f"| {gap_str} | {format_val(tmean, 3)} | {ks_str} |"
            )
            lines.append(row)

        if det:
            # For deterministic: show match status
            py_v = r["py_mean"]
            jl_v = r["jl_mean"]
            if py_v is not None and jl_v is not None:
                match = "MATCH" if abs(py_v - jl_v) < 0.5 else "MISMATCH"
                lines.append(f"| **{match}** | | | | | | | | | {py_v} vs {jl_v} | | |")

    return "\n".join(lines)


def generate_text_report(results):
    lines = []
    lines.append("=" * 80)
    lines.append("CROSS-LANGUAGE VALIDATION REPORT")
    lines.append("=" * 80)
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for r in results:
        lines.append("-" * 80)
        lines.append(f"{r['instance']} (n={r['n']}, opt={r['opt']}) — {r['algorithm']} "
                      f"({'deterministic' if r['deterministic'] else 'stochastic'})")
        lines.append("")

        for lang, lang_name in [("py", "Python"), ("jl", "Julia")]:
            count = r[f"{lang}_count"]
            mean = r[f"{lang}_mean"]
            std = r[f"{lang}_std"]
            mn = r[f"{lang}_min"]
            mx = r[f"{lang}_max"]
            gap = r[f"{lang}_gap"]
            tmean = r[f"{lang}_time_mean"]

            if mean is None:
                lines.append(f"  {lang_name}: no results")
                continue

            gap_str = f" (gap={gap:.1f}%)" if gap is not None else ""
            lines.append(f"  {lang_name}: {mean:.1f} ± {std:.1f} [{mn:.0f}-{mx:.0f}]{gap_str} "
                          f"(n={count}, time={tmean:.3f}s)")

        if r["ks_pvalue"] is not None:
            lines.append(f"  KS test p-value: {r['ks_pvalue']:.4f}")
        elif r["deterministic"]:
            py_v = r["py_mean"]
            jl_v = r["jl_mean"]
            if py_v is not None and jl_v is not None:
                match = "MATCH" if abs(py_v - jl_v) < 0.5 else "MISMATCH"
                lines.append(f"  Deterministic: {match} (py={py_v:.0f}, jl={jl_v:.0f})")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-language validation for ROAR-NET TSP")
    parser.add_argument("--instances", default=None,
                        help="Comma-separated instance names (default: all)")
    parser.add_argument("--algorithms", default=None,
                        help="Comma-separated algorithm names (default: all)")
    parser.add_argument("--budget", type=float, default=0,
                        help="Time budget in seconds (0 = auto based on instance size)")
    parser.add_argument("--trials", type=int, default=100,
                        help="Number of trials for stochastic algorithms (default: 100)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run only 1 trial per algorithm (quick check)")
    parser.add_argument("--no-download", action="store_true",
                        help="Skip downloading TSPLIB instances")
    parser.add_argument("--output", "-o", default=None,
                        help="Save report to file")
    args = parser.parse_args()

    # ---- Resolve instances ----
    if args.instances:
        inst_names = [s.strip() for s in args.instances.split(",")]
    else:
        inst_names = list(INSTANCES.keys())

    # ---- Resolve algorithms ----
    if args.algorithms:
        algo_names = [s.strip() for s in args.algorithms.split(",")]
    else:
        algo_names = list(ALGORITHMS.keys())

    # ---- Resolve trial count ----
    trials = 1 if args.dry_run else args.trials

    print(f"Cross-Language Validation")
    print(f"  Instances: {', '.join(inst_names)}")
    print(f"  Algorithms: {', '.join(algo_names)}")
    print(f"  Trials: {'1 (dry run)' if args.dry_run else args.trials}")
    budget_display = f"{args.budget}s" if args.budget and args.budget > 0 else "auto (based on instance size)"
    print(f"  Budget: {budget_display}")
    print()

    # ---- Ensure instances ----
    ensure_instances(allow_download=not args.no_download, instance_names=inst_names)

    # ---- Run comparisons ----
    all_results = []
    for inst_name in inst_names:
        if inst_name not in INSTANCES:
            print(f"Unknown instance: {inst_name}")
            continue

        inst_info = INSTANCES[inst_name]
        inst_path = INSTANCES_DIR / inst_info["file"]
        if not inst_path.exists():
            print(f"Instance file not found: {inst_path}")
            continue

        for algo_name in algo_names:
            if algo_name not in ALGORITHMS:
                print(f"Unknown algorithm: {algo_name}")
                continue

            algo_info = ALGORITHMS[algo_name]
            n_trials = 1 if algo_info["deterministic"] else trials

            print(f"[{inst_name}] [{algo_name}] "
                  f"{'deterministic' if algo_info['deterministic'] else f'{n_trials} trials'}...",
                  end=" ", flush=True)

            result = compare_algorithm(
                inst_path, inst_name, inst_info,
                algo_name, algo_info,
                args.budget, n_trials,
            )

            py_mean = result["py_mean"]
            jl_mean = result["jl_mean"]

            if algo_info["deterministic"]:
                if py_mean is not None and jl_mean is not None:
                    match = abs(py_mean - jl_mean) < 0.5
                    status = "MATCH" if match else "MISMATCH"
                    print(f"py={py_mean:.0f} jl={jl_mean:.0f} [{status}]")
                else:
                    print("INCOMPLETE")
            else:
                pval = result["ks_pvalue"]
                if pval is not None:
                    print(f"py={py_mean:.1f} jl={jl_mean:.1f} (p={pval:.4f})")
                else:
                    print(f"py={py_mean:.1f} jl={jl_mean:.1f}")

            all_results.append(result)

    # ---- Generate report ----
    print()
    print("=" * 80)

    md_report = generate_markdown_report(all_results)
    text_report = generate_text_report(all_results)

    print(text_report)

    if args.output:
        with open(args.output, "w") as f:
            f.write(md_report)
        print(f"Markdown report saved to {args.output}")

    # Save markdown report to tests directory by default
    report_path = PROJECT_ROOT / "tests" / "cross_validation_report.md"
    with open(report_path, "w") as f:
        f.write(md_report)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
